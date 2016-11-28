#include "luaT.h"
#include "THC.h"

#include "utils.c"

//#include "BilinearSamplerBHWD.cu"
#include "BilinearSamplerPerspective.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcuptn(lua_State *L);

int luaopen_libcuptn(lua_State *L)
{
  lua_newtable(L);
  //cunn_BilinearSamplerBHWD_init(L);

  //lua_newtable(L);
  cunn_BilinearSamplerPerspective_init(L);
  return 1;
}

