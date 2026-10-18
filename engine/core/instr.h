#ifndef INSTR_H
#define INSTR_H

#include <stddef.h>

/* Instrumentation: which weight tensors does a forward pass touch, and how
   often? This is the data Phase 3's hot/cold eviction policy is designed
   around (see Engine-Design spec, component 2).
   Enable with env CP_INSTR=<path> — appends CSV rows:
     tensor,layer,token,bytes
   Costs one branch when unset, one fprintf per touch when set. */
void instr_touch(const char *tensor, int layer, int token, size_t bytes);
void instr_report(void); /* row-count summary to stderr + flush/close */

#endif
