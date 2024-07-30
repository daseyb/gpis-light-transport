#ifndef AFFINEARITHMETIC_HPP_
#define AFFINEARITHMETIC_HPP_

#include "Eigen/Dense"
#include "math/Vec.hpp"
#include "math/MathUtil.hpp"
#include <cfloat>

namespace Tungsten {


template<size_t d, typename Scalar>
struct AffineRange {
    Eigen::Array<Scalar, d, 1> lower, upper;

    Eigen::Array<Scalar, d, 1> width() {
        return upper - lower;
    }
};

enum struct RangeBound {
    Unknown,
    Negative,
    Positive
};

template<size_t d, typename Scalar=double>
struct Affine {

    enum struct Mode {
        AffineFixed,
        AffineTruncate,
        AffineAll,
    };

    using VecT = Eigen::Array<Scalar, d,  1>;
    using MatT = Eigen::Array<Scalar, d, -1>;

    VecT base;
    MatT aff;
    VecT err;

    static const Mode mode = Mode::AffineFixed;
    static const size_t n_keep = 40;

    Affine(Vec<Scalar,d> base, std::vector<Vec<Scalar, d>> aff = {}, Vec<Scalar, d> err = Vec<Scalar, d>(0.)) {
        this->aff.resize(d, aff.size());
        for (size_t i = 0; i < d; i++) {
            this->base.row(i) = base[i];
            this->err.row(i) = err[i];

            for (size_t j = 0; j < aff.size(); j++) {
                this->aff(i, j) = aff[j][i];
            }
        }
    }

    Affine(double base = 0., std::vector<double> aff = {}, double err = 0.) {
        this->base = VecT(base);
        this->aff = Eigen::Map<MatT>(aff.data(), 1, aff.size());
        this->err = VecT(err);
    }

    Affine(VecT base, MatT aff = MatT::Zero(d,0), VecT err = VecT::Zero(d)) : base(base), aff(aff), err(err) { }

    bool isConst() const {
        return err.isZero() && aff.cols() == 0;
    }

    VecT radius() const {
        if (isConst()) return VecT::Zero(base.rows());
        VecT res = err;
        for (size_t i = 0; i < aff.cols(); i++) {
            res += aff.col(i).cwiseAbs();
        }
        return res;
    }

    AffineRange<d, Scalar> mayContainBounds() const {
        auto rad = radius();
        return { base - rad, base + rad };
    }

    //https://github.com/nmwsharp/neural-implicit-queries/blob/main/src/affine.py#L127C11-L127C11
    Affine truncate() const {
        Affine result = *this;

        if (isConst() || mode != Mode::AffineTruncate) {
            return result;
        }

        if (result.aff.cols() < n_keep) {
            return result;
        }

        // Sort in decreasing magnitude
        std::vector<VecT> aff(result.aff.cols());
        for (size_t i = 0; i < result.aff.cols(); i++) {
            aff.push_back(result.aff.col(i));
        }

        std::sort(aff.begin(), aff.end(), [](auto a, auto b) { return a.cwiseAbs().sum() > b.cwiseAbs().sum(); });

        for (size_t i = n_keep; i < aff.size(); i++) {
            result.err += aff[i].cwiseAbs();
        }

        result.aff.resize(result.aff.rows(), n_keep);
        for (size_t i = 0; i < result.aff.cols(); i++) {
            result.aff.col(i) = aff[i];
        }

        return result;
    }

    Affine applyLinearApprox(const VecT& alpha, const VecT& beta, const VecT& delta) const {
        Affine result = *this;
        result.base = alpha * result.base + beta;

        result.aff *= alpha.replicate(1, result.aff.cols());


        switch (result.mode) {
        case Mode::AffineFixed:
            result.err = alpha * result.err + delta.cwiseAbs();
            return result;
        case Mode::AffineTruncate:
        case Mode::AffineAll:
            result.err = alpha * result.err;

            Eigen::MatrixXd deltaMat = delta.cwiseAbs().matrix().asDiagonal();
            MatT deltaAr = deltaMat.array();
            MatT affResult(result.aff.rows(), result.aff.cols() + deltaMat.cols());
            affResult << result.aff, deltaAr;
            result.aff = affResult;

            result = result.truncate();
            return result;
        }
    }

    Affine<1> operator[](size_t i) const {
        return Affine<1>(base.row(i), aff.row(i), err.row(i));
    }

    Affine<1,Scalar> x() const {
        return (*this)[0];
    }

    Affine<1, Scalar> y() const {
        return (*this)[1];
    }

    Affine<1, Scalar> z() const {
        return (*this)[2];
    }

    Affine operator-() const
    {
        Affine result(-base, -aff, err);
        return result;
    }

    static std::pair<Affine, Affine> longer_shorter(const Affine& a, const Affine& b) {
        if (a.aff.cols() > b.aff.cols()) {
            return { a,b };
        }
        else {
            return { b,a };
        }
    }

    Affine operator+(const Affine& other) const
    {
        auto [longer, shorter] = longer_shorter(*this, other);
        longer.base += shorter.base;
        for (unsigned i = 0; i < shorter.aff.cols(); ++i)
            longer.aff.col(i) += shorter.aff.col(i);
        longer.err += shorter.err;
        return longer;
    }

    Affine operator-(const Affine& other) const
    {
        Affine result = *this;
        result.base -= other.base;
        result.aff.resize(result.aff.rows(), std::max(result.aff.cols(), other.aff.cols()));

        for (unsigned i = 0; i < other.aff.cols(); ++i)
            result.aff.col(i) -= other.aff.col(i);

        result.err += other.err;
        return result;
    }

    Affine operator*(const Affine& other) const
    {
        auto [a, b] = longer_shorter(*this, other);
        Affine result(a.base * b.base, a.aff, a.err);

        result.aff.resize(Eigen::NoChange, result.aff.cols() + 1);

        for (unsigned i = 0; i < result.aff.cols()-1; ++i) {
            auto a_aff_i = a.aff.col(i);
            auto b_aff_i = (i < b.aff.cols()) ? (VecT)b.aff.col(i) : VecT::Zero(a.aff.rows());
            result.aff.col(i) = a.base * b_aff_i + b.base * a_aff_i;
        }

        result.aff.col(result.aff.cols() - 1) = a.radius() * b.radius();
        return result.truncate();
    }


    Affine operator+(const double& a) const
    {
        Affine result = *this;
        result.base += a;
        return result;
    }

    Affine operator-(const double& a) const
    {
        Affine result = *this;
        result.base -= a;
        return result;
    }

    Affine operator*(const double& a) const
    {
        Affine result = *this;
        result.base *= a;
        result.aff *= a;
        result.err *= abs(a);
        return result;
    }

    Affine operator/(const double& a) const
    {
        Affine result = *this;
        result.base /= a;
        result.aff /= a;
        result.err /= abs(a);
        return result;
    }

    Affine operator+=(const Affine& other)
    {
        auto [longer, shorter] = longer_shorter(*this, other);
        longer.base += shorter.base;
        for (unsigned i = 0; i < shorter.aff.cols(); ++i)
            longer.aff.col(i) += shorter.aff.col(i);
        longer.err += shorter.err;
        *this = longer;
        return *this;
    }

    Affine operator-=(const Affine& other)
    {
        Affine result = *this;
        result.base -= other.base;
        result.aff.resize(result.aff.rows(), std::max(result.aff.cols(), other.aff.cols()));

        for (unsigned i = 0; i < other.aff.cols(); ++i)
            result.aff.col(i) -= other.aff.col(i);

        result.err += other.err;
        *this = result;
        return result;
    }

    Affine pow2() const {
        Affine result(base * base, aff, err);

        result.aff.resize(Eigen::NoChange, result.aff.cols() + 1);

        for (unsigned i = 0; i < result.aff.cols() - 1; ++i) {
            auto a_aff_i =  aff.col(i);
            result.aff.col(i) = 2 * base * a_aff_i;
        }

        auto rad = radius();
        result.aff.col(result.aff.cols() - 1) = rad * rad * 0.5;
        result.base += rad * rad * 0.5;
        return result.truncate();
    }

    Affine<1> lengthSq() const {
        Affine<1> result(0.);
        for (int i = 0; i < d; i++) {
            Affine<1> el = (*this)[i];
            auto resEl = el;
            resEl.aff.resize(Eigen::NoChange, std::max(result.aff.cols(), resEl.aff.cols()));
            resEl.aff.setZero();
            resEl.aff.block(0, 0, 1, el.aff.cols()) = el.aff;
            result += el.pow2();
        }
        return result;
    }

    Affine<1> length() const {
        return sqrt(lengthSq());
    }

    RangeBound rangeBound() const {
        static_assert(d == 1, "Range bounds only implemented for 1D affines.");
        auto r = mayContainBounds();
        if (r.lower(0) > 0) return RangeBound::Positive;
        else if (r.upper(0) < 0) return RangeBound::Negative;
        else return RangeBound::Unknown;
    }
};



template<size_t d, typename Scalar>
Affine<d, Scalar> sqrt(const Affine<d, Scalar>& x) {
    if (x.isConst()) {
        return Affine<d, Scalar>{ sqrt(x.base) };
    }
    auto i = x.mayContainBounds();
    i.lower = i.lower.cwiseMax(0.);
    auto sq = AffineRange<d,Scalar>{ sqrt(i.lower), sqrt(i.upper) };
    auto c = (sq.upper + sq.lower).cwiseMax(0.000000001);
    auto h = sq.upper - sq.lower;
    auto alpha = 1.0 / c;
    auto dzeta = c / 8.0 + 0.5 * sq.lower * sq.upper / c;
    auto delta = h * h / (8.0 * c);

    auto la = x.applyLinearApprox(alpha, dzeta, delta);

    auto mask = (i.upper < 0);

    return Affine<d, Scalar>(
        mask.select(decltype(la.base)::Zero(), la.base),
        mask.replicate(1, la.aff.cols()).select(decltype(la.aff)::Zero(la.aff.rows(), la.aff.cols()), la.aff),
        mask.select(decltype(la.base)::Zero(), la.err)
    );
}


template<size_t d, typename Scalar>
typename Affine<d, Scalar>::VecT find_alpha2(const Affine<d, Scalar>& P) {
    auto bx = P.mayContainBounds();
    auto bslope = cosBound(bx);
    typename Affine<d, Scalar>::VecT alpha = 0.5 * (bslope.lower + bslope.upper);
    return alpha.cwiseMax(-1).cwiseMin(1);
}


template<size_t d, typename Scalar>
Affine<d, Scalar> aff_sin(const Affine<d, Scalar>& x) {
    if (x.isConst()) {
        return Affine<d, Scalar>(sin(x.base));
    }

    auto bx = x.mayContainBounds();

    typename Affine<d, Scalar>::VecT alpha = find_alpha2(x);

    auto intA = acos(alpha);
    auto intB = -intA;

    auto first = [lower = bx.lower](auto x) { return 2. * PI * ceil((lower + x) / (2. * PI)) - x; };
    auto last = [upper = bx.upper](auto x) { return 2. * PI * floor((upper - x) / (2. * PI)) + x; };

    typename Affine<d, Scalar>::VecT extremes[] = {
        bx.lower, bx.upper, first(intA), last(intA), first(intB), last(intB)
    };

    typename Affine<d, Scalar>::VecT r_lower;
    r_lower.resizeLike(bx.lower);
    r_lower = DBL_MAX;

    typename Affine<d, Scalar>::VecT r_upper;
    r_upper.resizeLike(bx.upper);
    r_upper = -DBL_MAX;

    for (auto& ex : extremes) {
        ex = ex.cwiseMax(bx.lower).cwiseMin(bx.upper);
        auto ey = sin(ex) - alpha * ex;

        r_lower = r_lower.cwiseMin(ey);
        r_upper = r_upper.cwiseMax(ey);
    }

    Eigen::ArrayXd beta = 0.5 * (r_upper + r_lower);
    Eigen::ArrayXd delta = r_upper - beta;

    return x.applyLinearApprox(alpha, beta, delta);
}

template<size_t d, typename Scalar>
AffineRange<d, Scalar> sinBound(AffineRange<d, Scalar> x) {
    auto f_lower = sin(x.lower);
    auto f_upper = sin(x.upper);

    // test if there is an interior peak in the range
    x.lower /= 2. * PI;
    x.upper /= 2. * PI;
    auto contains_min = ceil(x.lower - .75) < (x.upper - .75);
    auto contains_max = ceil(x.lower - .25) < (x.upper - .25);

    // result is either at enpoints or maybe an interior peak
    decltype(x.lower) out_lower = contains_min.select(-decltype(x.lower)::Ones(x.lower.rows()), f_lower.cwiseMin(f_upper));
    decltype(x.lower) out_upper = contains_max.select(decltype(x.lower)::Ones(x.lower.rows()), f_lower.cwiseMax(f_upper));

    return { out_lower, out_upper };
}


template<size_t d, typename Scalar>
AffineRange<d, Scalar> cosBound(AffineRange<d, Scalar> x) {
    return sinBound(AffineRange<d, Scalar>{x.lower + PI / 2, x.upper + PI / 2 });
}


template<size_t d, typename Scalar>
Affine<d, Scalar> aff_cos(const Affine<d, Scalar>& x) {
    return aff_sin(x + PI / 2);
}

template<size_t d, typename Scalar>
Affine<1, Scalar> dot(const Eigen::Vector<Scalar,d>& a, const Affine<d, Scalar>& b) {
    Affine<1, Scalar> result;
    for (int i = 0; i < d; i++) {
        result += b[i] * a[i];
    }
    return result;
}

template<size_t d, typename Scalar>
Affine<1, Scalar> dot(const Vec<Scalar,unsigned(d)> & a, const Affine<d, Scalar>& b) {
    Affine<1, Scalar> result;
    for (int i = 0; i < d; i++) {
        result += b[i] * a[i];
    }
    return result;
}



}

#endif /* AFFINEARITHMETIC_HPP_ */
