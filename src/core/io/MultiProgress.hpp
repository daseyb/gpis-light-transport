#include <atomic>
#include <mutex>
#include <functional>
#include <array>
#include <iostream>

namespace Tungsten {

class ProgressBar {
public:
    void set_progress(float value) {
        std::unique_lock lock{ mutex_ };  // CTAD (C++17)
        progress_ = value;
    }

    void set_bar_width(size_t width) {
        std::unique_lock lock{ mutex_ };
        bar_width_ = width;
    }

    void fill_bar_progress_with(const std::string& chars) {
        std::unique_lock lock{ mutex_ };
        fill_ = chars;
    }

    void fill_bar_remainder_with(const std::string& chars) {
        std::unique_lock lock{ mutex_ };
        remainder_ = chars;
    }

    void set_status_text(const std::string& status) {
        std::unique_lock lock{ mutex_ };
        status_text_ = status;
    }

    void update(float value, std::ostream& os = std::cout) {
        set_progress(value);
        write_progress(os);
    }

    void write_progress(std::ostream& os = std::cout) {
        std::unique_lock lock{ mutex_ };

        // No need to write once progress is 100%
        if (progress_ > 1.0f) return;

        // Move cursor to the first position on the same line and flush 
        os << "\r" << std::flush;

        // Start bar
        os << "[";

        const auto completed = static_cast<size_t>(progress_ * static_cast<float>(bar_width_));
        for (size_t i = 0; i < bar_width_; ++i) {
            if (i <= completed)
                os << fill_;
            else
                os << remainder_;
        }

        // End bar
        os << "]";

        // Write progress percentage
        os << " " << std::min(static_cast<size_t>(progress_*100.f), size_t(100)) << "%";

        // Write status text
        os << " " << status_text_;
    }

private:
    std::mutex mutex_;
    float progress_{ 0.0f };

    size_t bar_width_{ 60 };
    std::string fill_{ "#" }, remainder_{ " " }, status_text_{ "" };
};

template <typename Indicator, size_t count>
class MultiProgress {
public:
    template <size_t index>
    typename std::enable_if<(index >= 0 && index < count), void>::type
        update(float value, std::ostream& os = std::cout) {
        bars_[index].set_progress(value);
        write_progress(os);
    }

    void update(size_t index, float value, std::ostream& os = std::cout) {
        bars_[index].set_progress(value);
        write_progress(os);
    }

    void write_progress(std::ostream& os = std::cout) {
        std::unique_lock lock{ mutex_ };

        // Move cursor up if needed
        if (started_)
            for (size_t i = 0; i < count; ++i)
                os << "\x1b[A";

        // Write each bar
        for (auto& bar : bars_) {
            bar.write_progress();
            os << "\n";
        }

        if (!started_)
            started_ = true;
    }

private:
    std::array<Indicator, count> bars_;
    std::mutex mutex_;
    std::atomic<bool> started_{ false };
};
}