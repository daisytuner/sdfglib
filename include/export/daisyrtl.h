#ifndef __DAISY_RTL_H__
#define __DAISY_RTL_H__

#ifdef __DAISY_NVVM__

// type conversion
#define __daisy_d2i_hi __nvvm_d2i_hi
#define __daisy_d2i_lo __nvvm_d2i_lo
#define __daisy_lohi_i2d __nvvm_lohi_i2d

#define __daisy_d2i_rn __nvvm_d2i_rn
#define __daisy_d2i_rm __nvvm_d2i_rm
#define __daisy_d2i_rp __nvvm_d2i_rp
#define __daisy_d2i_rz __nvvm_d2i_rz

#define __daisy_i2d_rn __nvvm_i2d_rn
#define __daisy_i2d_rm __nvvm_i2d_rm
#define __daisy_i2d_rp __nvvm_i2d_rp
#define __daisy_i2d_rz __nvvm_i2d_rz

#define __daisy_d2f_rn __nvvm_d2f_rn
#define __daisy_d2f_rm __nvvm_d2f_rm
#define __daisy_d2f_rp __nvvm_d2f_rp
#define __daisy_d2f_rz __nvvm_d2f_rz

#define __daisy_d2ui_rn __nvvm_d2ui_rn
#define __daisy_d2ui_rm __nvvm_d2ui_rm
#define __daisy_d2ui_rp __nvvm_d2ui_rp
#define __daisy_d2ui_rz __nvvm_d2ui_rz

#define __daisy_ui2d_rn __nvvm_ui2d_rn
#define __daisy_ui2d_rm __nvvm_ui2d_rm
#define __daisy_ui2d_rp __nvvm_ui2d_rp
#define __daisy_ui2d_rz __nvvm_ui2d_rz

#define __daisy_d2ll_rn __nvvm_d2ll_rn
#define __daisy_d2ll_rm __nvvm_d2ll_rm
#define __daisy_d2ll_rp __nvvm_d2ll_rp
#define __daisy_d2ll_rz __nvvm_d2ll_rz

#define __daisy_ll2d_rn __nvvm_ll2d_rn
#define __daisy_ll2d_rm __nvvm_ll2d_rm
#define __daisy_ll2d_rp __nvvm_ll2d_rp
#define __daisy_ll2d_rz __nvvm_ll2d_rz

#define __daisy_d2ull_rn __nvvm_d2ull_rn
#define __daisy_d2ull_rm __nvvm_d2ull_rm
#define __daisy_d2ull_rp __nvvm_d2ull_rp
#define __daisy_d2ull_rz __nvvm_d2ull_rz

#define __daisy_ull2d_rn __nvvm_ull2d_rn
#define __daisy_ull2d_rm __nvvm_ull2d_rm
#define __daisy_ull2d_rp __nvvm_ull2d_rp
#define __daisy_ull2d_rz __nvvm_ull2d_rz

#define __daisy_f2i_rn __nvvm_f2i_rn
#define __daisy_f2i_rm __nvvm_f2i_rm
#define __daisy_f2i_rp __nvvm_f2i_rp
#define __daisy_f2i_rz __nvvm_f2i_rz

#define __daisy_i2f_rn __nvvm_i2f_rn
#define __daisy_i2f_rm __nvvm_i2f_rm
#define __daisy_i2f_rp __nvvm_i2f_rp
#define __daisy_i2f_rz __nvvm_i2f_rz

#define __daisy_f2ui_rn __nvvm_f2ui_rn
#define __daisy_f2ui_rm __nvvm_f2ui_rm
#define __daisy_f2ui_rp __nvvm_f2ui_rp
#define __daisy_f2ui_rz __nvvm_f2ui_rz

#define __daisy_ui2f_rn __nvvm_ui2f_rn
#define __daisy_ui2f_rm __nvvm_ui2f_rm
#define __daisy_ui2f_rp __nvvm_ui2f_rp
#define __daisy_ui2f_rz __nvvm_ui2f_rz

#define __daisy_f2ll_rn __nvvm_f2ll_rn
#define __daisy_f2ll_rm __nvvm_f2ll_rm
#define __daisy_f2ll_rp __nvvm_f2ll_rp
#define __daisy_f2ll_rz __nvvm_f2ll_rz

#define __daisy_ll2f_rn __nvvm_ll2f_rn
#define __daisy_ll2f_rm __nvvm_ll2f_rm
#define __daisy_ll2f_rp __nvvm_ll2f_rp
#define __daisy_ll2f_rz __nvvm_ll2f_rz

#define __daisy_f2ull_rn __nvvm_f2ull_rn
#define __daisy_f2ull_rm __nvvm_f2ull_rm
#define __daisy_f2ull_rp __nvvm_f2ull_rp
#define __daisy_f2ull_rz __nvvm_f2ull_rz

#define __daisy_ull2f_rn __nvvm_ull2f_rn
#define __daisy_ull2f_rm __nvvm_ull2f_rm
#define __daisy_ull2f_rp __nvvm_ull2f_rp
#define __daisy_ull2f_rz __nvvm_ull2f_rz

#define __daisy_f2bf16_rn __nvvm_f2bf16_rn
#define __daisy_f2bf16_rz __nvvm_f2bf16_rz

#define __daisy_f2h_rn __nvvm_f2h_rn

// saturate
#define __daisy_saturate_f __nvvm_saturate_f
#define __daisy_saturate_d __nvvm_saturate_d

// fma instructions
#define __daisy_fma_rn_f __nvvm_fma_rn_f
#define __daisy_fma_rn_d __nvvm_fma_rn_d

#define __daisy_fma_rm_f __nvvm_fma_rm_f
#define __daisy_fma_rm_d __nvvm_fma_rm_d

#define __daisy_fma_rp_f __nvvm_fma_rp_f
#define __daisy_fma_rp_d __nvvm_fma_rp_d

#define __daisy_fma_rz_f __nvvm_fma_rz_f
#define __daisy_fma_rz_d __nvvm_fma_rz_d

#define __daisy_fma_rn_ftz_f __nvvm_fma_rn_ftz_f

#define __daisy_fma_rm_ftz_f __nvvm_fma_rm_ftz_f

#define __daisy_fma_rp_ftz_f __nvvm_fma_rp_ftz_f

#define __daisy_fma_rz_ftz_f __nvvm_fma_rz_ftz_f

#endif  // __DAISY_NVVM__

#endif  // __DAISY_RTL_H__
