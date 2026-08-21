/* Minimal ARMCI stubs so we can link ScaFaCoS without GlobalArrays.
 * Needed only because libfcs_fmm.so references these; the P3M path does not
 * call them at runtime.
 */
void armci_die(char *msg, int code) {
  (void)msg;
  (void)code;
}

long atomic_fetch_and_add(long *ptr, long val) {
  long old = *ptr;
  *ptr += val;
  return old;
}
