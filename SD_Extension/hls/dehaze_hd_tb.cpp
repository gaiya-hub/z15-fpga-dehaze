#include "dehaze_hd.h"
#include "../firmware/board_neural_v3.h"
#include "../firmware/board_statistics_fast.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
static void read(const std::string &name,void *p,int bytes){std::ifstream f(name,std::ios::binary);f.read((char*)p,bytes);if(!f)throw std::runtime_error(name);}
int main(){uint32_t state=8;for(int k=0;k<100000;k++){state=state*1664525U+1013904223U;if(nn_fast_sqrt(state)!=nn_isqrt(state))return 2;}
std::ifstream cases("fixtures/cases.txt");int w,h,ar,ag,ab,sr,sg,sb,blend,errors=0;
while(cases>>w>>h>>ar>>ag>>ab>>sr>>sg>>sb>>blend){int n=w*h;std::string name="fixtures/"+std::to_string(w)+"x"+std::to_string(h);
std::vector<rgb_pixel> input(n),out(n),mem(256+16*n+3168);std::vector<unsigned char> work(2*n+18*((w+3)/4)*((h+3)/4)+64);std::vector<uint32_t> orig(n),base(n),expected(n),aux(n),fields(10*n),globals(22),temporary(n),cmem(256+11*n),cpu(n);
read(name+".input",orig.data(),4*n);read(name+".base",base.data(),4*n);read(name+".expected",expected.data(),4*n);read(name+".aux",aux.data(),4*n);read(name+".features",fields.data(),40*n);read(name+".global",globals.data(),88);
for(int i=0;i<n;i++)input[i]=orig[i];
dehaze_hd(input.data(),out.data(),work.data(),w,h,ar,ag,ab,160,96,64,2,sr,sg,sb,blend,mem.data());
for(int i=0;i<n;i++){if((uint32_t)out[i]!=base[i]||(uint32_t)mem[256+i]!=aux[i]){if(errors<6)std::cerr<<"DCP/aux mismatch "<<name<<" "<<i<<"\n";errors++;}cmem[256+i]=(uint32_t)mem[256+i];}
nn_prepare(orig.data(),base.data(),cmem.data(),w,h,temporary.data());
std::vector<uint32_t> fast(256+11*n);for(int i=0;i<n;i++)fast[256+i]=aux[i];
nn_prepare_fast(orig.data(),base.data(),fast.data(),w,h);
for(int i=0;i<10*n;i++)if(fast[256+n+i]!=fields[i]){if(errors<6)std::cerr<<"FAST field mismatch "<<name<<" "<<i<<"\n";errors++;}
for(int i=0;i<n;i++)if(fast[256+i]!=base[i])errors++;

for(int i=0;i<10*n;i++)if(cmem[256+n+i]!=fields[i]){if(errors<6)std::cerr<<"feature mismatch "<<name<<" "<<i<<" "<<cmem[256+n+i]<<" "<<fields[i]<<"\n";errors++;}
std::vector<uint32_t> network(3168);read(name+".network",network.data(),12672);for(int j=0;j<3168;j++)mem[256+11*n+j]=network[j];for(int i=0;i<22;i++){mem[i]=globals[i];cmem[i]=globals[i];}mem[22]=NN_TAG;cmem[22]=NN_TAG;
for(int i=0;i<10*n;i++)mem[256+n+i]=cmem[256+n+i];
nn_cpu_infer(cpu.data(),cmem.data(),n);
dehaze_hd(input.data(),out.data(),work.data(),w,h,ar,ag,ab,160,96,64,3,sr,sg,sb,blend,mem.data());
for(int i=0;i<n;i++)if((uint32_t)out[i]!=expected[i]||cpu[i]!=expected[i]){if(errors<6)std::cerr<<"NN mismatch "<<name<<" "<<i<<" hw="<<std::hex<<(uint32_t)out[i]<<" cpu="<<cpu[i]<<" expected="<<expected[i]<<std::dec<<"\n";errors++;}
if((uint32_t)mem[23]!=NN_TAG)errors++;std::cout<<"CASE "<<name<<" completed\n";}
std::cout<<(errors?"FAIL ":"PASS ")<<errors<<" integer mismatches\n";return errors?1:0;}
