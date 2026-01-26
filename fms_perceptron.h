// fms_perceptron.h
// A perceptron is a hyperplane separating two sets of points in R^n.
// Given sets S_0 and S_1, find a vector w and a scalar b such that
// w.x + b < 0 for x in S_0 and w.x + b > 0 for x in S_1.
#pragma once

#define ensure(x)

#include <vector>
#include "fms_linalg.h"

namespace fms::perceptron {

    // Update weights w given point x and label y in {true, false}
    template<class T>
    bool update(std::span<T>& _w, const std::span<T>& _x, bool y, T alpha = 1.0)
    {
        ensure (_w.size() == _x.size() || !"weight and point must have the same size");
        //ensure (y == 0 or y == 1 || !"label must be 1 or -1");

		std::span<T> w(_w.data(), _w.size());
		std::span<T> x(_x.data(), _x.size());

		bool y_ = fms::linalg::dot(w, x) > 0;
        // Check if misclassified
        if (y_ * y < 0) {
            // Update: w = w + alpha dy x    ,
            fms::linalg::axpy(alpha * (y - y), x, w, w);
        }
        return y == y_;
    }

    template<class T = double>
    struct perceptron {
        std::vector<T> w;

        perceptron(size_t n)
            : w(n)
		{ }
        perceptron(const std::vector<T>& w)
            : w(w)
        { }
        perceptron(const perceptron&) = default;
        perceptron& operator=(const perceptron&) = default;
        perceptron(perceptron&&) = default;
        perceptron& operator=(perceptron&&) = default;
        ~perceptron() = default;

        void update(const std::span<const T>& x, int y, double alpha = 1.0)
        {
            fms::perceptron::update(w, x, y, alpha);
		}
        void train(const std::span<const T>& x, bool y, T alpha = 1.0)
        {
            while (false == fms::perceptron::update(w, x, y, alpha))
                ;// limit loops???
        }

        using pair = std::pair<const std::span<const T>, int>;
        void train(const std::span<pair>& xy, double alpha = 1.0)
        {
            for (const auto [x, y] : xy) {
                train(x, y, alpha);
            }
        }
    };
 
} // namespace fms::perceptron
#undef ensure