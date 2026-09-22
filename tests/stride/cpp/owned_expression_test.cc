#include "execute/update.h"
namespace ffi=xla::ffi;
#include "thread_pool_test_support.h"
namespace n=tensor0::stride;
namespace s=n::scalar;
void Finish(testing::ThreadPool& pool,ffi::Future future) {
  bool ready=false;future.OnReady([&](const std::optional<ffi::Error>& error){assert(!error);ready=true;});
  if(!ready)pool.run_parallel();
  assert(ready);
}
int main(){
  constexpr uint64_t size=65536;
  for(uint64_t batches:{1,4}) {
    testing::ThreadPool pool(4);
    auto record=n::layout::BuildLayout({3},{-1},5,{2},2,size,size,0);
    std::vector<double> source(size*batches);std::vector<float> base(size*batches,3),result(size*batches,-1);
    for(uint64_t i=0;i<source.size();++i)source[i]=(i%19)+0.25;
    double alpha=2,beta=3;
    using SourceMap=n::expression::Cast<s::F32,n::expression::Identity<s::F64>>;
    using BaseMap=n::expression::Scale<s::F64,n::expression::Identity<s::F32>>;
    Finish(pool,n::ExecuteUpdate<s::F32,s::F64,s::F64>(pool.get(),{record},source.data(),base.data(),result.data(),size,size,batches,&alpha,1,&beta,1,SourceMap{},BaseMap{2,{}}));
    for(uint64_t b=0;b<batches;++b)for(uint64_t i=0;i<size;++i){
      float expected=(i==2||i==4||i==6)?2*static_cast<float>(source[b*size+5-(i-2)/2])+18:3;
      assert(result[b*size+i]==expected);
    }
    std::vector<std::complex<float>> cs(size*batches,{2,1}),cb(size*batches,{3,-1}),cr(size*batches);
    float a=2,z=3;
    Finish(pool,n::ExecuteUpdate<s::C64,s::F32,s::F32>(pool.get(),{record},cs.data(),cb.data(),cr.data(),size,size,batches,&a,1,&z,1,n::expression::Conjugate<s::C64>{},n::expression::Identity<s::C64>{}));
    for(uint64_t i=0;i<cr.size();++i)assert(cr[i]==((i%size==2||i%size==4||i%size==6)?std::complex<float>(13,-5):cb[i]));
  }
}
