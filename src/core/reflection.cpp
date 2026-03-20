#include "core/reflection.h"

#include <iostream>



namespace refl
{

const char* TypeTagToString(const TypeTag tag)
{
#define CASE(lbl) case TypeTag::lbl: return #lbl

	switch (tag)
	{
		CASE(Unknown);
		CASE(Bool);
		CASE(Int8);
		CASE(UInt8);
		CASE(Int16);
		CASE(UInt16);
		CASE(Int32);
		CASE(UInt32);
		CASE(Int64);
		CASE(UInt64);
		CASE(Half);
		CASE(Float);
		CASE(Double);
		CASE(LongDouble);
		CASE(Pointer);
		CASE(ConstPointer);
		CASE(Structure);
		CASE(Count);

		default:
			EM_WTF("Unhandled TypeTag enumerant");
			return "";
	}

#undef CASE
}

void DumpTypeInfo(const TypeMetaInfo* pMetaInfo)
{
	for (const TypeMetaInfo* info = pMetaInfo; info; info = info->next)
	{
		std::cout << "name=" << info->name << ", "
				  << "offset=" << info->offset << ", "
				  << "size=" << info->size << ", "
				  << "location=" << info->location << ", "
				  << "count=" << info->count << ", "
				  << "type=" << TypeTagToString(info->type) << std::endl;

		if (info->type == TypeTag::Structure)
			DumpTypeInfo(info->fields);
	}
}

} // End of namespace refl
