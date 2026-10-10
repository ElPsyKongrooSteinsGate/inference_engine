#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <omp.h>

struct Config {
    int dim = 0, hidden_dim = 0, n_layers = 0, n_heads = 0;
    int n_kv_heads = 0, vocab_size = 0, seq_len = 0;
};

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int> s) : shape(std::move(s)) {
        size_t n = 1;
        for (int d : shape) {
            if (d <= 0) throw std::runtime_error("Invalid tensor dimension");
            n *= static_cast<size_t>(d);
        }
        data.assign(n, 0.0f);
    }
    size_t size() const { return data.size(); }
    float& operator()(int r, int c) { return data[static_cast<size_t>(r) * shape[1] + c]; }
    const float& operator()(int r, int c) const { return data[static_cast<size_t>(r) * shape[1] + c]; }
};

struct QuantizedTensorINT8 {
    std::vector<int8_t> data;
    std::vector<int> shape;
    float scale = 1.0f;

    // Checkpoint matrices are stored row-major as [out_dim, in_dim].
    void quantize(const float* src, int out_dim, int in_dim) {
        shape = {out_dim, in_dim};
        const size_t n = static_cast<size_t>(out_dim) * in_dim;
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; ++i) max_abs = std::max(max_abs, std::abs(src[i]));
        scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = 1.0f / scale;
        data.resize(n);
        for (size_t i = 0; i < n; ++i) {
            int q = static_cast<int>(std::round(src[i] * inv));
            q = std::clamp(q, -127, 127);
            data[i] = static_cast<int8_t>(q);
        }
    }

    static float reduce_add(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        return _mm_cvtss_f32(lo);
    }

    static Tensor matmul(const Tensor& A, const QuantizedTensorINT8& B) {
        if (A.shape.size() != 2 || B.shape.size() != 2 || A.shape[1] != B.shape[1])
            throw std::runtime_error("matmul shape mismatch");
        const int M = A.shape[0], K = A.shape[1], N = B.shape[0];
        Tensor C({M, N});
        #pragma omp parallel for collapse(2) schedule(static) if(static_cast<long long>(M) * N >= 4096)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                const float* a = A.data.data() + static_cast<size_t>(i) * K;
                const int8_t* b = B.data.data() + static_cast<size_t>(j) * K;
                __m256 acc = _mm256_setzero_ps();
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 av = _mm256_loadu_ps(a + k);
                    int64_t packed;
                    std::memcpy(&packed, b + k, sizeof(packed));
                    __m128i b8 = _mm_cvtsi64_si128(packed);
                    __m256i b32 = _mm256_cvtepi8_epi32(b8);
                    __m256 bv = _mm256_cvtepi32_ps(b32);
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                float sum = reduce_add(acc);
                for (; k < K; ++k) sum += a[k] * static_cast<float>(b[k]);
                C.data[static_cast<size_t>(i) * N + j] = sum * B.scale;
            }
        }
        return C;
    }
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static void softmax(float* x, int n) {
    if (n <= 0) return;
    float m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - m); sum += x[i]; }
    if (sum > 0.0f) for (int i = 0; i < n; ++i) x[i] /= sum;
}

static int sample_next_token(const std::vector<float>& logits, std::mt19937& rng,
                             const std::unordered_map<int,int>& counts,
                             float temperature, int top_k, float repetition_penalty) {
    if (logits.empty()) return 0;
    temperature = std::max(temperature, 1e-6f);
    top_k = std::clamp(top_k, 1, static_cast<int>(logits.size()));
    std::vector<std::pair<float,int>> ranked;
    ranked.reserve(logits.size());
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        float x = logits[i];
        auto it = counts.find(i);
        if (it != counts.end() && it->second > 0 && repetition_penalty > 1.0f)
            x = x >= 0.0f ? x / std::pow(repetition_penalty, it->second)
                          : x * std::pow(repetition_penalty, it->second);
        ranked.emplace_back(x, i);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + top_k, ranked.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    float mx = ranked[0].first;
    std::vector<double> weights;
    std::vector<int> ids;
    weights.reserve(top_k); ids.reserve(top_k);
    for (int i = 0; i < top_k; ++i) {
        ids.push_back(ranked[i].second);
        weights.push_back(std::exp((ranked[i].first - mx) / temperature));
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return ids[dist(rng)];
}

struct KVCache {
    int max_seq_len = 0, d_model = 0, current_pos = 0;
    Tensor k_cache, v_cache;
    KVCache() = default;
    KVCache(int max_len, int dim) : max_seq_len(max_len), d_model(dim),
        k_cache({max_len, dim}), v_cache({max_len, dim}) {}
    bool append(const Tensor& k, const Tensor& v) {
        int n = k.shape[0];
        if (current_pos + n > max_seq_len) return false;
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < d_model; ++d) {
                k_cache(current_pos + i, d) = k(i, d);
                v_cache(current_pos + i, d) = v(i, d);
            }
        current_pos += n;
        return true;
    }
};

class RMSNorm {
public:
    int dim;
    float eps = 1e-5f;
    std::vector<float> weight;
    explicit RMSNorm(int d = 1) : dim(d), weight(d, 1.0f) {}
    Tensor forward(const Tensor& x) const {
        Tensor y({x.shape[0], dim});
        for (int i = 0; i < x.shape[0]; ++i) {
            float ss = 0.0f;
            for (int d = 0; d < dim; ++d) ss += x(i,d) * x(i,d);
            float inv = 1.0f / std::sqrt(ss / dim + eps);
            for (int d = 0; d < dim; ++d) y(i,d) = x(i,d) * inv * weight[d];
        }
        return y;
    }
};

// llama2.c uses split-half rotary pairs: i and i + head_dim/2.
static void apply_rope(Tensor& token, int pos, int head_dim) {
    int heads = token.shape[1] / head_dim;
    int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        int base = h * head_dim;
        for (int i = 0; i < half; ++i) {
            float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2*i) / head_dim);
            float angle = pos * freq, c = std::cos(angle), s = std::sin(angle);
            int a = base + i, b = base + i + half;
            float x0 = token(0,a), x1 = token(0,b);
            token(0,a) = x0*c - x1*s;
            token(0,b) = x0*s + x1*c;
        }
    }
}

class QuantizedMultiHeadAttention {
public:
    int d_model, num_heads, head_dim;
    QuantizedTensorINT8 W_q, W_k, W_v, W_o;
    QuantizedMultiHeadAttention(int d, int h) : d_model(d), num_heads(h), head_dim(d/h) {}

    Tensor forward(const Tensor& X, KVCache& cache, int start_pos) {
        int n = X.shape[0];
        Tensor Q = QuantizedTensorINT8::matmul(X, W_q);
        Tensor K = QuantizedTensorINT8::matmul(X, W_k);
        Tensor V = QuantizedTensorINT8::matmul(X, W_v);
        for (int i = 0; i < n; ++i) {
            Tensor q({1,d_model}), k({1,d_model});
            for (int d = 0; d < d_model; ++d) { q(0,d)=Q(i,d); k(0,d)=K(i,d); }
            apply_rope(q, start_pos+i, head_dim);
            apply_rope(k, start_pos+i, head_dim);
            for (int d = 0; d < d_model; ++d) { Q(i,d)=q(0,d); K(i,d)=k(0,d); }
        }
        if (!cache.append(K,V)) throw std::runtime_error("KV cache overflow");
        int total = cache.current_pos;
        Tensor out({n,d_model});
        const float att_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int h = 0; h < num_heads; ++h) {
            int off = h * head_dim;
            Tensor scores({n,total});
            for (int i = 0; i < n; ++i) {
                int qpos = total - n + i;
                for (int j = 0; j < total; ++j) {
                    if (j > qpos) { scores(i,j) = -1e9f; continue; }
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += Q(i,off+d) * cache.k_cache(j,off+d);
                    scores(i,j) = dot * att_scale;
                }
                softmax(scores.data.data() + static_cast<size_t>(i)*total, total);
            }
            for (int i = 0; i < n; ++i)
                for (int d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < total; ++j)
                        sum += scores(i,j) * cache.v_cache(j,off+d);
                    out(i,off+d) = sum;
                }
        }
        return QuantizedTensorINT8::matmul(out, W_o);
    }
};

class QuantizedSwiGLUFFN {
public:
    int d_model, hidden_dim;
    QuantizedTensorINT8 W_gate, W_up, W_down;
    QuantizedSwiGLUFFN(int d, int h) : d_model(d), hidden_dim(h) {}
    Tensor forward(const Tensor& x) const {
        Tensor gate = QuantizedTensorINT8::matmul(x, W_gate);
        Tensor up = QuantizedTensorINT8::matmul(x, W_up);
        Tensor mid({x.shape[0], hidden_dim});
        for (int i = 0; i < x.shape[0]; ++i)
            for (int d = 0; d < hidden_dim; ++d)
                mid(i,d) = silu(gate(i,d)) * up(i,d);
        return QuantizedTensorINT8::matmul(mid, W_down);
    }
};

class QuantizedTransformerBlock {
public:
    RMSNorm attn_norm, ffn_norm;
    QuantizedMultiHeadAttention attention;
    QuantizedSwiGLUFFN ffn;
    QuantizedTransformerBlock(int d, int heads, int hidden)
        : attn_norm(d), ffn_norm(d), attention(d,heads), ffn(d,hidden) {}
    Tensor forward(const Tensor& x, KVCache& cache, int pos) {
        Tensor norm1 = attn_norm.forward(x);
        Tensor a = attention.forward(norm1, cache, pos);
        Tensor residual({x.shape[0], x.shape[1]});
        for (size_t i=0; i<x.size(); ++i) residual.data[i] = x.data[i] + a.data[i];
        Tensor norm2 = ffn_norm.forward(residual);
        Tensor f = ffn.forward(norm2);
        Tensor out({x.shape[0], x.shape[1]});
        for (size_t i=0; i<out.size(); ++i) out.data[i] = residual.data[i] + f.data[i];
        return out;
    }
};

class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::unordered_map<std::string,int> ids;

    bool load(const std::string& path, int vocab_size) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t max_len = 0;
        if (!f.read(reinterpret_cast<char*>(&max_len), sizeof(max_len)) ||
            vocab_size <= 0 || max_len > (1u<<20)) return false;
        vocab.clear(); scores.clear(); ids.clear();
        vocab.reserve(vocab_size); scores.reserve(vocab_size);
        for (int i=0; i<vocab_size; ++i) {
            float score; uint32_t len;
            if (!f.read(reinterpret_cast<char*>(&score), sizeof(score)) ||
                !f.read(reinterpret_cast<char*>(&len), sizeof(len)) ||
                len > max_len) return false;
            std::string s(len, '\0');
            if (len && !f.read(s.data(), static_cast<std::streamsize>(len))) return false;
            scores.push_back(score); ids[s]=i; vocab.push_back(std::move(s));
        }
        return true;
    }
    int find(const std::string& s) const {
        auto it=ids.find(s); return it==ids.end() ? -1 : it->second;
    }
    std::string decode(int id) const {
        if (id==1 || id==2 || id<0 || id>=static_cast<int>(vocab.size())) return "";
        std::string s=vocab[id];
        const std::string marker="\xE2\x96\x81";
        size_t p=0;
        while ((p=s.find(marker,p)) != std::string::npos) { s.replace(p,marker.size()," "); ++p; }
        return s;
    }
    // Approximate score-based BPE. For faithful results, replace with SentencePiece.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out{1};
        if (text.empty()) return out;
        std::string input="\xE2\x96\x81"+text;
        std::vector<std::string> pieces;
        for (unsigned char c : input) pieces.emplace_back(1, static_cast<char>(c));
        while (pieces.size()>1) {
            float best=-std::numeric_limits<float>::infinity();
            int best_i=-1, best_id=-1;
            for (int i=0; i+1<static_cast<int>(pieces.size()); ++i) {
                int id=find(pieces[i]+pieces[i+1]);
                if (id>=0 && scores[id]>best) { best=scores[id]; best_i=i; best_id=id; }
            }
            if (best_i<0) break;
            pieces[best_i]=vocab[best_id];
            pieces.erase(pieces.begin()+best_i+1);
        }
        for (const auto& p : pieces) {
            int id=find(p);
            if (id>=0) out.push_back(id);
            else for (unsigned char c : p) {
                std::string b(1,static_cast<char>(c));
                int bi=find(b);
                if (bi>=0) out.push_back(bi);
            }
        }
        return out;
    }
};

class Stories110MEngine {
public:
    Config config;
    int actual_vocab_size=0;
    bool shared_weights=false;
    Tensor embeddings;
    std::vector<QuantizedTransformerBlock> layers;
    RMSNorm final_norm;
    QuantizedTensorINT8 lm_head;
    std::vector<KVCache> caches;

    bool load_model(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { std::cerr << "Cannot open model: " << path << "\n"; return false; }
        std::streamoff file_size=f.tellg();
        f.seekg(0);
        int h[7];
        if (!f.read(reinterpret_cast<char*>(h), sizeof(h))) return false;
        config={h[0],h[1],h[2],h[3],h[4],h[5],h[6]};
        if (config.vocab_size==0 || config.dim<=0 || config.hidden_dim<=0 ||
            config.n_layers<=0 || config.n_heads<=0 || config.n_kv_heads<=0 ||
            config.seq_len<=0 || config.dim%config.n_heads!=0 ||
            config.n_kv_heads!=config.n_heads) {
            std::cerr << "Invalid/unsupported checkpoint configuration\n";
            return false;
        }
        actual_vocab_size=std::abs(config.vocab_size);
        shared_weights=config.vocab_size>0;
        auto read_exact=[&](void* dst,size_t bytes) {
            return static_cast<bool>(f.read(static_cast<char*>(dst),static_cast<std::streamsize>(bytes)));
        };
        auto read_floats=[&](size_t n) {
            std::vector<float> v(n);
            if (!read_exact(v.data(), n*sizeof(float))) throw std::runtime_error("Truncated checkpoint");
            return v;
        };
        std::cout << "Model: dim=" << config.dim << " hidden=" << config.hidden_dim
                  << " layers=" << config.n_layers << " heads=" << config.n_heads
                  << " vocab=" << actual_vocab_size << " context=" << config.seq_len << "\n";
        final_norm=RMSNorm(config.dim);
        embeddings=Tensor({actual_vocab_size,config.dim});
        if (!read_exact(embeddings.data.data(),embeddings.size()*sizeof(float))) return false;
        layers.reserve(config.n_layers);
        caches.reserve(config.n_layers);
        for (int l=0;l<config.n_layers;++l) {
            layers.emplace_back(config.dim,config.n_heads,config.hidden_dim);
            caches.emplace_back(config.seq_len,config.dim);
            auto& b=layers.back();
            if (!read_exact(b.attn_norm.weight.data(),config.dim*sizeof(float))) return false;
            auto q=read_floats(static_cast<size_t>(config.dim)*config.dim);
            auto k=read_floats(static_cast<size_t>(config.dim)*config.dim);
            auto v=read_floats(static_cast<size_t>(config.dim)*config.dim);
            auto o=read_floats(static_cast<size_t>(config.dim)*config.dim);
            b.attention.W_q.quantize(q.data(),config.dim,config.dim);
            b.attention.W_k.quantize(k.data(),config.dim,config.dim);
            b.attention.W_v.quantize(v.data(),config.dim,config.dim);
            b.attention.W_o.quantize(o.data(),config.dim,config.dim);
            if (!read_exact(b.ffn_norm.weight.data(),config.dim*sizeof(float))) return false;
            auto gate=read_floats(static_cast<size_t>(config.hidden_dim)*config.dim);
            auto down=read_floats(static_cast<size_t>(config.dim)*config.hidden_dim);
            auto up=read_floats(static_cast<size_t>(config.hidden_dim)*config.dim);
            b.ffn.W_gate.quantize(gate.data(),config.hidden_dim,config.dim);
            b.ffn.W_down.quantize(down.data(),config.dim,config.hidden_dim);
            b.ffn.W_up.quantize(up.data(),config.hidden_dim,config.dim);
        }
        if (!read_exact(final_norm.weight.data(),config.dim*sizeof(float))) return false;

        // Legacy llama2.c checkpoints place two RoPE tables before an untied classifier.
        const size_t head_dim=static_cast<size_t>(config.dim/config.n_heads);
        const size_t rope_bytes=2 * static_cast<size_t>(config.seq_len) *
                                (head_dim/2) * sizeof(float);
        std::streamoff pos=f.tellg();
        if (pos<0 || pos>file_size) return false;
        size_t remaining=static_cast<size_t>(file_size-pos);
        size_t head_bytes=shared_weights ? 0 :
            static_cast<size_t>(actual_vocab_size)*config.dim*sizeof(float);
        if (remaining == rope_bytes + head_bytes) {
            f.seekg(static_cast<std::streamoff>(rope_bytes),std::ios::cur);
        } else if (remaining != head_bytes) {
            std::cerr << "Unexpected checkpoint size/layout after final norm\n";
            return false;
        }
        if (!shared_weights) {
            auto head=read_floats(static_cast<size_t>(actual_vocab_size)*config.dim);
            lm_head.quantize(head.data(),actual_vocab_size,config.dim);
        } else {
            lm_head.quantize(embeddings.data.data(),actual_vocab_size,config.dim);
        }
        std::cout << "Loaded and quantized weights to INT8.\n";
        return true;
    }

    std::vector<float> forward(const std::vector<int>& ids, int pos) {
        if (ids.empty()) throw std::runtime_error("Empty token batch");
        if (pos<0 || pos+static_cast<int>(ids.size())>config.seq_len)
            throw std::runtime_error("Input exceeds context length");
        Tensor x({static_cast<int>(ids.size()),config.dim});
        for (int i=0;i<static_cast<int>(ids.size());++i) {
            if (ids[i]<0 || ids[i]>=actual_vocab_size) throw std::runtime_error("Token ID out of range");
            for (int d=0;d<config.dim;++d) x(i,d)=embeddings(ids[i],d);
        }
        for (int l=0;l<config.n_layers;++l) x=layers[l].forward(x,caches[l],pos);
        x=final_norm.forward(x);
        Tensor last({1,config.dim});
        for (int d=0;d<config.dim;++d) last(0,d)=x(x.shape[0]-1,d);
        return QuantizedTensorINT8::matmul(last,lm_head).data;
    }
};

int main(int argc,char** argv) {
    try {
        std::string model_path="stories110M.bin", tokenizer_path="tokenizer.bin";
        std::string prompt="Once upon a time";
        int max_tokens=40, top_k=25;
        float temperature=0.85f;
        for (int i=1;i<argc;++i) {
            std::string a=argv[i];
            auto need=[&]() -> std::string {
                if (i+1>=argc) throw std::runtime_error("Missing value after "+a);
                return argv[++i];
            };
            if (a=="-i" || a=="-input" || a=="--input") prompt=need();
            else if (a=="-n" || a=="-num_tokens" || a=="--tokens") max_tokens=std::stoi(need());
            else if (a=="-t" || a=="-temp" || a=="--temp") temperature=std::stof(need());
            else if (a=="-k" || a=="-top_k" || a=="--top_k") top_k=std::stoi(need());
            else if (a=="-m" || a=="--model") model_path=need();
            else if (a=="-z" || a=="--tokenizer") tokenizer_path=need();
            else throw std::runtime_error("Unknown argument: "+a);
        }
        if (max_tokens<0) throw std::runtime_error("Token count must be non-negative");
        Stories110MEngine model;
        if (!model.load_model(model_path)) return 1;
        Tokenizer tokenizer;
        if (!tokenizer.load(tokenizer_path,model.actual_vocab_size)) {
            std::cerr << "Could not load tokenizer: " << tokenizer_path << "\n";
            return 1;
        }
        auto prompt_ids=tokenizer.encode(prompt);
        if (prompt_ids.empty()) throw std::runtime_error("Tokenizer returned no tokens");
        if (static_cast<int>(prompt_ids.size())>=model.config.seq_len) {
            std::cerr << "Prompt is too long for the model context\n"; return 1;
        }
        std::cout << "\nPrompt: " << prompt << "\nGenerated: " << std::flush;
        std::mt19937 rng(1234);
        std::unordered_map<int,int> counts;
        for (int id:prompt_ids) ++counts[id];
        auto start=std::chrono::steady_clock::now();
        auto logits=model.forward(prompt_ids,0);
        int pos=static_cast<int>(prompt_ids.size());
        int generated=0;
        int next=sample_next_token(logits,rng,counts,temperature,top_k,1.25f);
        while (generated<max_tokens && pos<model.config.seq_len && next!=1 && next!=2) {
            std::cout << tokenizer.decode(next) << std::flush;
            ++counts[next]; ++generated;
            if (generated>=max_tokens || pos>=model.config.seq_len) break;
            logits=model.forward({next},pos++);
            next=sample_next_token(logits,rng,counts,temperature,top_k,1.25f);
        }
        double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout << "\n\nInference: " << secs << " s, "
                  << (secs>0 ? generated/secs : 0.0) << " generated tokens/s ("
                  << generated << " tokens)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
