
// fms_linalg.h - Generic linear algebra utilities
#pragma once
#include <numeric>
#include <span>
#include <stdexcept>

namespace fms::linalg {

	// Compute the dot product of two vectors
	template<class T>
	constexpr T dot(std::size_t n, T* x, T* y)
	{
		return std::inner_product(x, x + n, y, T(0));
	}
	// Prefer span to pointers
	template<class T>
	constexpr T dot(std::span<T> x, std::span<T> y)
	{
		if (x.size() != y.size()) {
			throw std::runtime_error("dot: size mismatch");
		}
		return std::inner_product(x.begin(), x.end(), y.begin(), T(0));
	}
	namespace {
		constexpr bool test_dot() {
			constexpr double a1[] = { 1.0, 2.0, 3.0 };
			constexpr double a2[] = { 4.0, 5.0, 6.0 };
			std::span<const double> s1(a1, 3);
			std::span<const double> s2(a2, 3);
			// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
			return dot(s1, s2) == 32.0;
		}
		static_assert(test_dot(), "dot product test failed");

	}
	
	// z = a * x + y
	template<class T>
	constexpr void axpy(T a, std::span<T> x, std::span<T> y, std::span<T> z)
	{
		for (size_t i = 0; i < z.size(); ++i) {
			z[i] = a * x[i] + y[i];
		}
	}

} // namespace fms::linalg	
