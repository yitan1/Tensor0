#include "execute/update.h"
namespace ffi=xla::ffi;
#include "thread_pool_test_support.h"
#include <sys/mman.h>
#include <unistd.h>
namespace n=tensor0::stride;
namespace s=n::scalar;
struct SnapshotMap : n::expression::Identity<s::F32> {
  float shift=0;
  std::shared_ptr<int> state=std::make_shared<int>(42);
  SnapshotMap()=default;
  SnapshotMap(const SnapshotMap&)=default;
  SnapshotMap(SnapshotMap&&){throw std::runtime_error("unexpected move");}
  float operator()(float x)const {assert(*state==42);return x+shift;}
};
void Lifetime() {
  constexpr uint64_t size=65536;
  for(uint64_t batches:{1,4}) {
    testing::ThreadPool pool(4);
    std::vector<float> source(size*batches,2),base(size*batches,3),result(size*batches,-1);
    float alpha=2,beta=3;bool ready=false;
    std::weak_ptr<int> source_life,base_life;
    {
      SnapshotMap source_map,base_map;source_map.shift=1;base_map.shift=2;
      source_life=source_map.state;base_life=base_map.state;
      const auto record=n::layout::BuildLayout({size},{1},0,{1},0,size,size,0);
      auto future=n::ExecuteUpdate<s::F32,s::F32,s::F32>(pool.get(),{record},source.data(),base.data(),result.data(),size,size,batches,&alpha,1,&beta,1,source_map,base_map);
      future.OnReady([&](const std::optional<ffi::Error>& e){assert(!e);ready=true;});
      assert(!ready);
      source_map.shift=100;base_map.shift=200;
    }
    assert(!source_life.expired()&&!base_life.expired());
    pool.run_parallel();assert(ready);
    assert(source_life.expired()&&base_life.expired());
    for(float x:result)assert(x==21);
  }
}
struct FailingMap : n::expression::Identity<s::F32> {
  bool unknown=false;
  float operator()(float)const {if(unknown)throw 42;throw std::runtime_error("map execution failure");}
};
void ExecutionFailure() {
  constexpr uint64_t size=65536,batches=4;
  for(bool unknown:{false,true}) {
    testing::ThreadPool pool(4);
    FailingMap map;map.unknown=unknown;
    std::vector<float> source(size*batches,2),base(size*batches,3),result(size*batches,-1);
    float alpha=2,beta=3;bool ready=false;
    auto record=n::layout::BuildLayout({1},{1},0,{1},0,size,size,0);
    auto future=n::ExecuteUpdate<s::F32,s::F32,s::F32>(pool.get(),{record},source.data(),base.data(),result.data(),size,size,batches,&alpha,1,&beta,1,map,n::expression::Identity<s::F32>{});
    future.OnReady([&](const std::optional<ffi::Error>& e){assert(e);ready=true;});
    pool.run_parallel();assert(ready);
    for(float x:result)assert(x==3);
  }
}
void GuardedSkip() {
  const std::size_t bytes=sysconf(_SC_PAGESIZE);
  void* guarded=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);assert(guarded!=MAP_FAILED);
  testing::ThreadPool pool;float base=3,result=-1,alpha=0,beta=2;
  auto record=n::layout::BuildLayout({1},{1},0,{1},0,1,1,0);
  auto error=testing::CompletedError(n::ExecuteUpdate<s::F32,s::F32,s::F32>(pool.get(),{record},static_cast<const float*>(guarded),&base,&result,1,1,1,&alpha,1,&beta,1,FailingMap{},n::expression::Identity<s::F32>{}));
  assert(error.success()&&result==6);assert(munmap(guarded,bytes)==0);
}
int main(){Lifetime();ExecutionFailure();GuardedSkip();}
