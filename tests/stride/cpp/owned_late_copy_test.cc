#include "execute/update.h"
namespace ffi=xla::ffi;
#include "thread_pool_test_support.h"
namespace n=tensor0::stride;
namespace s=n::scalar;
struct Control {bool fail=false;bool unknown=false;};
struct Map : n::expression::Identity<s::F32> {
  std::shared_ptr<Control> control=std::make_shared<Control>();
  Map()=default;
  Map(const Map& other):control(other.control) {
    if(control->fail){if(control->unknown)throw 42;throw std::runtime_error("late map copy");}
  }
};
int main(){
  constexpr uint64_t size=65536,batches=4;
  for(bool unknown:{false,true}) {
    testing::ThreadPool pool(4);Map map;map.control->unknown=unknown;
    std::vector<float> source(size*batches,2),base(size*batches,3),result(size*batches,-7);
    float alpha=2,beta=3;bool ready=false;
    auto record=n::layout::BuildLayout({1},{1},0,{1},0,size,size,0);
    auto future=n::ExecuteUpdate<s::F32,s::F32,s::F32>(pool.get(),{record},source.data(),base.data(),result.data(),size,size,batches,&alpha,1,&beta,1,map,n::expression::Identity<s::F32>{});
    future.OnReady([&](const std::optional<ffi::Error>& error){assert(error);ready=true;});
    assert(!ready);map.control->fail=true;pool.run_parallel();assert(ready);
    for(float value:result)assert(value==-7);
  }
}
