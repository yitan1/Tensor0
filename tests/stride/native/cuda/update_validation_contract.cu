// Native CUDA graph and grid-boundary contract for packed multi-record Update.
#include "cuda/update.cu"
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
namespace n = tensor0::stride::cuda::update;
namespace l = tensor0::stride::layout;
namespace d = tensor0::stride::descriptor;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void req(bool v, const char* s) { if (!v) throw std::runtime_error(s); }
template<class T> struct Device {
  T* p = nullptr;
  explicit Device(size_t n) { ck(cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(T))); }
  ~Device() { cudaFree(p); }
  void put(const std::vector<T>& v, cudaStream_t stream) { ck(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,stream)); }
  void get(std::vector<T>& v, cudaStream_t stream) { ck(cudaMemcpyAsync(v.data(),p,v.size()*sizeof(T),cudaMemcpyDeviceToHost,stream)); ck(cudaStreamSynchronize(stream)); }
};
struct Case {
  std::vector<int64_t> words, packed;
  std::unique_ptr<n::UpdatePreparedState> state;
  cudaStream_t stream;
  Device<float> source, base, result, alpha, beta;
  Device<int64_t> descriptor;
  int64_t source_dims[2], output_dims[2], coeff_dims[1], desc_dims[1];
  XLA_FFI_Buffer src{}, old{}, out{}, a{}, b{}, desc{};
  Case(std::vector<int64_t> w, int64_t batch)
    : words(std::move(w)), packed(l::PackOwnerFiber(d::DecodeLayout(words.data(),words.size()),1).words),
      source(batch*words[1]), base(batch*words[2]), result(batch*words[2]),
      alpha(batch), beta(batch), descriptor(packed.size()),
      source_dims{batch,words[1]},output_dims{batch,words[2]},coeff_dims{batch},desc_dims{static_cast<int64_t>(packed.size())} {
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto prepared=n::InstantiateUpdate({words.data(),words.size()});
    req(prepared.has_value(),"prepare"); state=std::move(*prepared);
    descriptor.put(packed,stream); ck(cudaStreamSynchronize(stream));
    auto buffer=[](void* ptr,XLA_FFI_DataType type,int64_t rank,int64_t* dims) { return XLA_FFI_Buffer{XLA_FFI_Buffer_STRUCT_SIZE,nullptr,type,ptr,rank,dims}; };
    src=buffer(source.p,XLA_FFI_DataType_F32,2,source_dims);
    old=buffer(base.p,XLA_FFI_DataType_F32,2,output_dims);
    out=buffer(result.p,XLA_FFI_DataType_F32,2,output_dims);
    a=buffer(alpha.p,XLA_FFI_DataType_F32,1,coeff_dims);
    b=buffer(beta.p,XLA_FFI_DataType_F32,1,coeff_dims);
    desc=buffer(descriptor.p,XLA_FFI_DataType_S64,1,desc_dims);
  }
  ~Case(){ cudaStreamDestroy(stream); }
  void invoke() {
    auto error=n::Update<ffi::F32,float,float>(state.get(),ffi::AnyBuffer(&src),ffi::BufferR2<ffi::F32>(&old),ffi::AnyBuffer(&a),ffi::AnyBuffer(&b),ffi::BufferR1<ffi::S64>(&desc),ffi::BufferR2<ffi::F32>(&out),stream);
    req(!error.failure(),"Update rejected");
  }
};
void graph() {
  Case c({1,600,600,4,1,6,10,7,-1,1,0,4,17,1,0,200,5,2,3,1,0,300,0,1,1},2);
  cudaGraph_t graph; cudaGraphExec_t exec;
  ck(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal)); c.invoke();
  ck(cudaStreamEndCapture(c.stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  for(int replay=0;replay<3;++replay) {
    std::vector<float> src(1200),base(1200),out(1200),aa(2),bb(2);
    for(int i=0;i<1200;++i) {src[i]=float((i%31)+replay*3);base[i]=float(40+(i%13)+replay);}
    aa={replay==1?0.f:2.f, replay==2?0.f:3.f};
    bb={replay==2?0.f:4.f,replay==1?0.f:5.f};
    if(replay==1) for(auto& x:src) x=NAN;
    if(replay==2) for(auto& x:base) x=NAN;
    c.source.put(src,c.stream); c.base.put(base,c.stream); c.alpha.put(aa,c.stream); c.beta.put(bb,c.stream);
    ck(cudaStreamSynchronize(c.stream));
    ck(cudaGraphLaunch(exec,c.stream)); c.result.get(out,c.stream);
    std::vector<float> expected=base;
    for(int batch=0;batch<2;++batch) {
      auto update=[&](int from,int to) {auto x=src[batch*600+from],y=base[batch*600+to]; expected[batch*600+to]=(aa[batch]==0?0:aa[batch]*x)+(bb[batch]==0?0:bb[batch]*y);};
      for(int i=0;i<7;++i) update(6-i,10+i);
      update(4,17);
      for(int i=0;i<5;++i) update(2*i,200+3*i);
    }
    for(size_t i=0;i<out.size();++i) req((std::isnan(expected[i])&&std::isnan(out[i])) || out[i]==expected[i],"graph output mismatch");
  }
  ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
  std::cout<<"graph 3 replay passed\n";
}
void alias() {
  Case c({1,16,16,2,1,1,1,3,1,1,1,9,9,3,1,1},2);
  c.src.data=c.old.data=c.out.data=c.base.p;
  std::vector<float> values(32),out(32);
  for(int i=0;i<32;++i) values[i]=float(i+1);
  c.base.put(values,c.stream); c.alpha.put(std::vector<float>{2,0},c.stream);
  c.beta.put(std::vector<float>{3,1},c.stream);
  c.invoke(); c.base.get(out,c.stream);
  for(int b=0;b<2;++b) for(int i=0;i<16;++i) {
    const bool touched=(i>=1&&i<=3)||(i>=9&&i<=11);
    req(out[b*16+i]==(touched&&b==0 ? 5*values[b*16+i] : values[b*16+i]),"alias mismatch");
  }
  std::cout<<"fused alias passed\n";
}
void fallback() {
  constexpr int64_t half=8388608, size=2*half;
  Case c({1,1,size,2,1,0,0,half,0,1,1,0,half,half,0,1},1);
  c.source.put(std::vector<float>{7},c.stream);
  c.base.put(std::vector<float>(size,3),c.stream);
  c.alpha.put(std::vector<float>{2},c.stream);
  c.beta.put(std::vector<float>{4},c.stream);
  ck(cudaStreamSynchronize(c.stream));
  c.invoke(); std::vector<float> out(size); c.result.get(out,c.stream);
  for(float x:out) req(x==26.f,"fallback output mismatch");
  std::cout<<"fallback 65536 blocks 2 records passed\n";
}
int main(int argc,char** argv) {
  try { if(argc>1) fallback(); else {graph(); alias();} return 0; }
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
