#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/BilinearSamplerPerspective.c"
//#include "generic/BilinearSamplerBHWD.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libptn(lua_State *L);
//LUA_EXTERNC DLL_EXPORT int luaopen_libptn_pp(lua_State *L);

int luaopen_libptn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "ptn");

  //nn_FloatBilinearSamplerBHWD_init(L);

  //nn_DoubleBilinearSamplerBHWD_init(L);

  nn_FloatBilinearSamplerPerspective_init(L);

  nn_DoubleBilinearSamplerPerspective_init(L);

  return 1;
}


