#pragma once

#include "core/global.h"
#include "core/meta.h"

#include <cstdint>


namespace refl
{

enum class TypeTag : uint32_t
{
	Unknown = 0,
	Bool,
	Int8,
	UInt8,
	Int16,
	UInt16,
	Int32,
	UInt32,
	Int64,
	UInt64,
	Half,
	Float,
	Double,
	LongDouble,
	Pointer,
	ConstPointer,
	Structure,
	Count
};

extern const char* TypeTagToString(const TypeTag tag);

template <typename T>
struct GetTypeTag
{
	static constexpr TypeTag value =
		std::is_class_v<T> && std::is_trivial_v<T> && std::is_standard_layout_v<T> ? TypeTag::Structure : TypeTag::Unknown;
};

template <>           struct GetTypeTag<bool>        { static constexpr TypeTag value = TypeTag::Bool;         };
template <>           struct GetTypeTag<int8_t>      { static constexpr TypeTag value = TypeTag::Int8;         };
template <>           struct GetTypeTag<uint8_t>     { static constexpr TypeTag value = TypeTag::UInt8;        };
template <>           struct GetTypeTag<int16_t>     { static constexpr TypeTag value = TypeTag::Int16;        };
template <>           struct GetTypeTag<uint16_t>    { static constexpr TypeTag value = TypeTag::UInt16;       };
template <>           struct GetTypeTag<int32_t>     { static constexpr TypeTag value = TypeTag::Int32;        };
template <>           struct GetTypeTag<uint32_t>    { static constexpr TypeTag value = TypeTag::UInt32;       };
template <>           struct GetTypeTag<int64_t>     { static constexpr TypeTag value = TypeTag::Int64;        };
template <>           struct GetTypeTag<uint64_t>    { static constexpr TypeTag value = TypeTag::UInt64;       };
template <>           struct GetTypeTag<float>       { static constexpr TypeTag value = TypeTag::Float;        };
template <>           struct GetTypeTag<double>      { static constexpr TypeTag value = TypeTag::Double;       };
template <>           struct GetTypeTag<long double> { static constexpr TypeTag value = TypeTag::LongDouble;   };
template <typename T> struct GetTypeTag<T*>          { static constexpr TypeTag value = TypeTag::Pointer;      };
template <typename T> struct GetTypeTag<const T*>    { static constexpr TypeTag value = TypeTag::ConstPointer; };


/* contains metadata about the member of a struct */
struct TypeMetaInfo
{
	const TypeMetaInfo* next;   //! links to the next field of the current structure
	const TypeMetaInfo* fields; //! for structures this links to type info of their fields, for other types this is NULL
	const char* name;           //! name of the member variable
	size_t offset;              //! its offset in the struct
	size_t size;                //! size of the property in bytes
	uint32_t location;          //! its position in the structure (i.e. defines its order)
	uint32_t count;             //! the number of elements for arrays or 0 in case of scalars and structures
	TypeTag type;               //! the type of the property
};

#define REFL_DECL_STRUCT_FIELD_(TYPE, NAME, ARRAY)                        \
	TYPE NAME ARRAY;                                                      \
	static inline const struct NAME ## Registrar                          \
	{                                                                     \
		const refl::TypeMetaInfo metaInfo;                                \
		template <typename T>                                             \
		static constexpr bool is_struct_v =                               \
			std::is_class_v<T> &&                                         \
			std::is_trivial_v<T> &&                                       \
			std::is_standard_layout_v<T>;                                 \
		template <typename T>                                             \
		static uint32_t GetLocation()                                     \
		{                                                                 \
			const uint32_t counter = sFieldCounter;                       \
			if constexpr (is_struct_v<T>)                                 \
				sFieldCounter += T::sFieldCounter;                        \
			else if constexpr (std::is_array_v<T>)                        \
				sFieldCounter += meta::get_array_element_count<T>::value; \
			else                                                          \
				sFieldCounter += 1;                                       \
			return counter;                                               \
		}                                                                 \
		template <typename T>                                             \
		static const refl::TypeMetaInfo* GetFieldList()                   \
		{                                                                 \
			if constexpr (is_struct_v<T>)                                 \
				return T::kMetaInfo;                                      \
			else                                                          \
				return nullptr;                                           \
		}                                                                 \
		NAME ## Registrar()                                               \
			: metaInfo {                                                  \
				kMetaInfo,                                                \
				GetFieldList<TYPE>(),                                     \
				#NAME,                                                    \
				offsetof(Type, NAME),                                     \
				sizeof(NAME),                                             \
				GetLocation<decltype(NAME)>(),                            \
				meta::get_array_element_count<decltype(NAME)>::value,     \
				refl::GetTypeTag<TYPE>::value                             \
			}                                                             \
		{                                                                 \
			kMetaInfo = &metaInfo;                                        \
		}                                                                 \
	} NAME ## _registrar_;

#define REFL_DECL_STRUCT_FIELD2(TYPE, NAME) REFL_DECL_STRUCT_FIELD_(TYPE, NAME, )
#define REFL_DECL_STRUCT_FIELD3(TYPE, NAME, ARRAY) REFL_DECL_STRUCT_FIELD_(TYPE, NAME, ARRAY)
#define REFL_DECL_STRUCT_FIELD(...) EM_CONCAT(REFL_DECL_STRUCT_FIELD, EM_NARGS(__VA_ARGS__))(__VA_ARGS__)

#define REFL_DECL_STRUCT_BEGIN(NAME)                                \
	struct NAME                                                     \
	{                                                               \
		typedef NAME Type;                                          \
		static inline uint32_t sFieldCounter = 0;                   \
		static inline const refl::TypeMetaInfo* kMetaInfo = nullptr;

#define REFL_DECL_STRUCT_END(NAME)                   \
		static inline const struct NAME ## Registrar \
		{                                            \
			const refl::TypeMetaInfo metaInfo;       \
			NAME ## Registrar()                      \
				: metaInfo {                         \
					nullptr,                         \
					kMetaInfo,                       \
					#NAME,                           \
					0,                               \
					sizeof(NAME),                    \
					0,                               \
					0,                               \
					refl::TypeTag::Structure         \
				}                                    \
			{                                        \
				kMetaInfo = &metaInfo;               \
			}                                        \
		} NAME ## _registrar_;                       \
	};

extern void DumpTypeInfo(const TypeMetaInfo* pMetaInfo);

template <typename T>
void DumpTypeInfo()
{
	DumpTypeInfo(T::kMetaInfo);
}

} // End of namespace refl
