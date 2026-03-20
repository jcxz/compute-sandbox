// This is a header to template meta programming building blocks and helpers

#pragma once

#include <cstddef>
#include <type_traits>


namespace meta
{

template <typename T>
struct get_array_element_count
{
	static constexpr size_t value = 0;
};

template <typename T, size_t N>
struct get_array_element_count<T[N]>
{
	static constexpr size_t value = N;
};

template <typename T, size_t N, size_t M>
struct get_array_element_count<T[N][M]>
{
	static constexpr size_t value = M * get_array_element_count<T[N]>::value;
};

} // End of namespace meta