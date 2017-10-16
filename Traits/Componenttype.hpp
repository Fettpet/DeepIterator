/**
 * \struct ComponentType
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief The ComponentType trait gives information about the type of the 
 * components of a container. You need to implement a specialication such that,
 * <b> typedef ComponentType< ContainerType >::type YourComponentType;</b>
 * is a valid and correct statement.
 * 
 */
#pragma once

namespace hzdr 
{
namespace details
{
struct UndefinedType;
}
namespace traits 
{
template<typename T>
struct ComponentType
{
    static_assert(true, "ComponentType wasnt specilized");
};

}//traits
}//hzdr
