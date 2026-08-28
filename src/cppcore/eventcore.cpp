// eventcore: native event-schedule core for the kvpim DAG engine.
// Experimental branch chenyi-822-cppcore-exp (chenyi9 ruling 2026-08-28).
//
// Scope v1: the SCHEDULING state machine only -- construction logic stays
// in Python; every created event is mirrored here as (device id, duration,
// dep indices).  This removes (a) the per-call dict copy of the old
// incremental scheduler (an accidental quadratic: finish/availability were
// rebuilt per decode admission round) and (b) the full-pass Python
// scheduler.  Semantics replicate _schedule_cacheblend exactly:
//   resource   = device (pipe) | one serial resource (no pipe)
//   start      = max(avail[resource], end[dep] for dep in deps)  (left fold)
//   end        = start + duration; avail[resource] = end
// Floats are IEEE doubles with the same fold order as the Python max(),
// so results are bit-identical.
#include <cstdint>
#include <vector>

namespace {
struct Core {
    int pipe = 0;
    std::vector<double> duration;
    std::vector<int32_t> device;
    std::vector<int32_t> dep_offset;   // size = n+1
    std::vector<int32_t> deps;
    std::vector<double> start_s;
    std::vector<double> end_s;
    std::vector<double> avail;         // by device id (or [0] when !pipe)
    int64_t advanced = 0;              // events scheduled so far
};
inline double res_avail(Core* c, int32_t dev) {
    size_t r = c->pipe ? static_cast<size_t>(dev) : 0;
    if (r >= c->avail.size()) c->avail.resize(r + 1, 0.0);
    return c->avail[r];
}
inline void res_set(Core* c, int32_t dev, double v) {
    size_t r = c->pipe ? static_cast<size_t>(dev) : 0;
    c->avail[r] = v;
}
}  // namespace

extern "C" {

void* ec_new(int pipe) {
    Core* c = new Core();
    c->pipe = pipe;
    c->dep_offset.push_back(0);
    return c;
}

void ec_free(void* h) { delete static_cast<Core*>(h); }

int64_t ec_size(void* h) {
    return static_cast<int64_t>(static_cast<Core*>(h)->duration.size());
}

// Returns the new event's index; deps must reference earlier indices.
int64_t ec_add(void* h, int32_t device, double duration,
               const int32_t* dep, int32_t ndep) {
    Core* c = static_cast<Core*>(h);
    int64_t index = static_cast<int64_t>(c->duration.size());
    for (int32_t i = 0; i < ndep; ++i) {
        if (dep[i] < 0 || dep[i] >= index) return -1;  // future/invalid dep
    }
    c->duration.push_back(duration);
    c->device.push_back(device);
    c->deps.insert(c->deps.end(), dep, dep + ndep);
    c->dep_offset.push_back(static_cast<int32_t>(c->deps.size()));
    return index;
}

void ec_set_duration(void* h, int64_t index, double duration) {
    Core* c = static_cast<Core*>(h);
    if (index >= 0 && index < static_cast<int64_t>(c->duration.size()))
        c->duration[index] = duration;
}

// Clear schedule state (keeps the event graph); next advance recomputes
// from scratch -- used after the W2 warm reprice.
void ec_reset(void* h) {
    Core* c = static_cast<Core*>(h);
    c->start_s.clear();
    c->end_s.clear();
    c->avail.assign(c->avail.size(), 0.0);
    c->advanced = 0;
}

// Schedule every event not yet scheduled.  Returns the count scheduled.
int64_t ec_advance(void* h) {
    Core* c = static_cast<Core*>(h);
    int64_t n = static_cast<int64_t>(c->duration.size());
    c->start_s.resize(n);
    c->end_s.resize(n);
    int64_t done = 0;
    for (int64_t i = c->advanced; i < n; ++i, ++done) {
        double start = res_avail(c, c->device[i]);
        for (int32_t k = c->dep_offset[i]; k < c->dep_offset[i + 1]; ++k) {
            double e = c->end_s[c->deps[k]];
            if (e > start) start = e;
        }
        double end = start + c->duration[i];
        c->start_s[i] = start;
        c->end_s[i] = end;
        res_set(c, c->device[i], end);
    }
    c->advanced = n;
    return done;
}

double ec_end(void* h, int64_t index) {
    Core* c = static_cast<Core*>(h);
    if (index < 0 || index >= c->advanced) return -1.0;
    return c->end_s[index];
}

// Bulk copy of the scheduled times; out arrays must hold ec_size entries.
void ec_bulk_times(void* h, double* out_start, double* out_end) {
    Core* c = static_cast<Core*>(h);
    for (size_t i = 0; i < c->start_s.size(); ++i) {
        out_start[i] = c->start_s[i];
        out_end[i] = c->end_s[i];
    }
}

}  // extern "C"
