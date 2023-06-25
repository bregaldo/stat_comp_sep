import torch
import pywph as pw
import string
import re


class DynamicalComputationDict(dict):
    """A dictionary-like object that computes symbolic keys on the fly."""

    def __init__(self):
        # Alphabet for the keys
        self.allowed_operations = ["*", "^"]
        self.allowed_functions_code = ["F", "IF", "IV", "SG", "A", "S", "M"] # FFT2, IFFT2, inverse, sign, absolute, sum, mean
        self.allowed_functions_code_mapping = ["F", "I", "V", "G", "A", "S", "M"] # We internally map allowed_functions_code elements to characters to simplify the parsing
        assert len(self.allowed_functions_code) == len(self.allowed_functions_code_mapping)
        assert len(set(self.allowed_functions_code_mapping)) == len(self.allowed_functions_code_mapping)

        # List of elementary operations
        self.allowed_functions = {
            "F": lambda x: torch.fft.fft2(x),
            "I": lambda x: torch.fft.ifft2(x),
            "V": lambda x: 1 / x,
            "G": lambda x: torch.sgn(x),
            "A": lambda x: torch.absolute(x),
            "S": lambda x: torch.sum(x, dim=(-1, -2)),
            "M": lambda x: torch.mean(x, dim=(-1, -2))
        }
        super().__init__()
    
    def __setitem__(self, key: str, item: str) -> None:
        raise AttributeError("DynamicalComputationDict items cannot be modified explicitly.")

    def __delitem__(self, key: str) -> None:
        raise AttributeError("DynamicalComputationDict items cannot be modified explicitly.")
    
    def _preprocess_key(self, key: str) -> str:
        # Check that key is well parenthesized
        paren = 0
        for c in key:
            if c == "(":
                paren += 1
            elif c == ")":
                paren -= 1
            if paren < 0:
                raise Exception(f"Key {key} is not well parenthesized.")
        if paren != 0:
            raise Exception(f"Key {key} is not well parenthesized.")
        
        # Replace allowed_functions_code elements by their mapping
        for i, elt in enumerate(self.allowed_functions_code):
            key = key.replace(elt, self.allowed_functions_code_mapping[i])
        
        # Check that the string contains only allowed characters
        key_copy = key
        for elt in self.allowed_functions_code_mapping + self.allowed_operations + list(string.ascii_lowercase) + list(string.digits) + ["(", ")"]:
            key_copy = key_copy.replace(elt, "")
        if key_copy != "":
            raise Exception(f"Key {key} contains invalid characters.")
    
        return key

    def _compute(self, key: str) -> None:
        # Initial parsing to replace expressions within parentheses by their value
        paren_start, paren_end, paren_cnt = None, None, 0
        subexps = []
        key_new = key
        for i, c in enumerate(key):
            if c == "(": # We have found the start of a subexpression
                paren_cnt += 1
                if paren_start is None:
                    paren_start = i
            elif c == ")":
                paren_cnt -= 1
                if paren_cnt == 0: # We have found the end of the subexpression
                    paren_end = i
                    subexp_key = key[paren_start+1:paren_end]
                    if subexp_key in self.keys():
                        subexps.append(super().__getitem__(subexp_key))
                    else:
                        subexps.append(self._compute(subexp_key))
                    key_new = key_new.replace(key[paren_start:paren_end+1], "#" + str(len(subexps) - 1), 1) # We replace the subexpression by its index in subexps
                    paren_start, paren_end = None, None
        
        # Parse the remaining string
        fns = ()
        curr_val = None
        waiting_for_nxtval = False # For the multiplication
        idx = 0
        while True:
            c = key_new[idx]
            if c in self.allowed_functions_code_mapping:
                fns += (self.allowed_functions[c],)
            elif c in string.ascii_lowercase:
                if c in self.var_list:
                    val = self[c]
                    if len(fns) > 0:
                        for fn in fns:
                            val = fn(val)
                        fns = ()
                    if curr_val is None:
                        assert not waiting_for_nxtval, f"Invalid key {key}."
                        curr_val = val
                    elif waiting_for_nxtval:
                        curr_val = curr_val * val
                        waiting_for_nxtval = False
                    else:
                        raise Exception(f"Invalid key {key}.")
                else:
                    raise Exception(f"Variable {c} not found.")
            elif c == "#":
                subexp_idx = [s for s in re.findall(r'\d+', key_new[idx+1:])][0]
                idx += len(subexp_idx)
                val = subexps[int(subexp_idx)]
                if len(fns) > 0:
                    for fn in fns:
                        val = fn(val)
                    fns = ()
                if curr_val is None:
                    assert not waiting_for_nxtval, f"Invalid key {key}."
                    curr_val = val
                elif waiting_for_nxtval:
                    curr_val = curr_val * val
                    waiting_for_nxtval = False
            elif c in self.allowed_operations:
                if c == "^":
                    if curr_val is None:
                        raise Exception(f"Invalid key {key}.")
                    else:
                        exp = [s for s in re.findall(r'\d+', key_new[idx+1:])][0]
                        idx += len(exp)
                        curr_val = curr_val ** int(exp)
                elif c == "*":
                    if curr_val is None:
                        raise Exception(f"Invalid key {key}.")
                    else:
                        waiting_for_nxtval = True
            else:
                raise Exception(f"Invalid character {c} in key {key}.")
            
            idx += 1
            if idx == len(key_new):
                break
        
        if curr_val is None:
            raise Exception(f"Invalid key {key}.")
        else:
            super().__setitem__(key, curr_val)
        return curr_val
        
    def __getitem__(self, key: str) -> torch.Tensor:
        key = self._preprocess_key(key)
        if key not in self.keys():
            self._compute(key)
        return super().__getitem__(key)

    def add_var(self, var: str, value: torch.Tensor) -> None:
        assert str.isalpha(var), "Variable name must be a string of letters."
        lvar = var.lower()
        if lvar in self.var_list:
            raise Exception(f"Variable {lvar} already exists.")
        else:
            self.var_list.append(lvar)
            super().__setitem__(lvar, value)
    
    def del_var(self, var: str) -> None:
        assert str.isalpha(var), "Variable name must be a string of letters."
        lvar = var.lower()
        if lvar not in self.var_list:
            raise Exception(f"Variable {lvar} does not exist.")
        else:
            self.var_list.remove(lvar)
            for key in list(self.keys()):
                if lvar in key:
                    t = super().__getitem__(key)
                    del t # To free memory occupied by the tensor
                    super().__delitem__(key)
    
    def update_var(self, var: str, value: torch.Tensor) -> None:
        assert str.isalpha(var), "Variable name must be a string of letters."
        lvar = var.lower()
        if lvar not in self.var_list:
            raise Exception(f"Variable {lvar} does not exist.")
        else:
            keys_to_update = [key for key in self.keys() if lvar in key and key != lvar]
            super().__setitem__(lvar, value)
            for key in keys_to_update:
                self.__getitem__(key)

class WaveletDict(DynamicalComputationDict):
    """A dictionary-like object that computes wavelet coefficients and any related derivated terms on the fly."""

    def __init__(self, M, N, J, L=4, device="cpu", dn=0, dj=None, precision='single'):
        assert dn == 0, "dn != 0 not implemented yet."

        self.M, self.N = M, N
        self.J, self.L = J, L
        self.dn, self.dj = dn, dj
        self.device = device
        self.precision = precision
        self.model = ["S11", "S00", "S01", "C01"] # Supported class of moments
        self.var_list = []

        # Load the WPH operator associated with the given parameters
        self.load_wph_op()

        super().__init__()
    
    def load_wph_op(self):
            self.wph_op = pw.WPHOp(self.M, self.N, self.J, L=self.L, dn=self.dn, device=self.device, precision=self.precision)
            self.wph_op.load_model(self.model, dj=self.dj)

            # Ghost preconfiguration to use the internal variables determined by the WPH operator
            self.wph_op.preconfigure(torch.zeros((self.M, self.N)), precompute_wt=False, precompute_modwt=False, nb_wph_cov_per_chunk=1e6)
            assert self.wph_op.nb_chunks_wph == 3 # Must be 3, first one for S11, second one for S00, third one for S01/C01

            # Useful internal variables of the WPH operator
            self._wph_moments_chunk_list = [x.clone() for x in self.wph_op.wph_moments_chunk_list]
            self._id_cov_indices = self.wph_op._id_cov_indices.clone()
            self._psi_1_indices = self.wph_op._psi_1_indices.clone()
            self._psi_2_indices = self.wph_op._psi_2_indices.clone()

            # For C01 calculations
            cov_chunk = self._wph_moments_chunk_list[2] # list id (nb_cov_chunk)
            c01_psi1_indices = self._psi_1_indices[cov_chunk] # (nb_cov_chunk) : id associated to (j1, t1)
            c01_psi2_indices =self._psi_2_indices[cov_chunk] # (nb_cov_chunk) : id associated to (j2, t2)
            sel = c01_psi1_indices != c01_psi2_indices # Selection for C01 moments
            self._c01_psi1_indices = c01_psi1_indices[sel]
            self._c01_psi2_indices = c01_psi2_indices[sel]

            # Add wavelets to the dictionary
            self.add_var('w', torch.fft.ifft2(self.wph_op.psi_f))

class WPHOp:
    def __init__(self,  M, N, J, L=4, device="cpu", dn=0, dj=None, precision='single'):
        self.wdict = WaveletDict(M, N, J, L=L, device=device, dn=dn, dj=dj, precision=precision)
        self.model = self.wdict.model
        self.Ntot = self.wdict.M * self.wdict.N
    
    def set_model(self, classes):
        for c in classes:
            assert c in ["S11", "S00", "S01", "C01"], f"Class {c} not supported."
        self.model = classes

    def compute_stats(self, x, derivatives=False, cov=None, fourier_cov=False):
        assert fourier_cov == False, "fourier_cov = True not implemented yet."
        assert cov is None or cov.shape == (self.wdict.M, self.wdict.N), "Covariance matrix must be of shape (M, N), representing a diagonal covariance matrix of the MxN pixels."

        # If no covariance matrix is provided, we assume a diagonal covariance matrix with unit variance
        if cov is None:
            cov = torch.ones((self.wdict.M, self.wdict.N), device=self.wdict.device)
        
        self.wdict.add_var('x', x.unsqueeze(-3))

        stats_list = []
        stats_grad_list, stats_hess_list = [], []
        for c in self.model:
            # Convenience strings
            wx_str = "IF((Fw)*(Fx))"
            awx_str = "A(IF((Fw)*(Fx)))"
            sgwx_str = f"SG({wx_str})"
            ivmwx_str = f"IV({awx_str})"

            wx = self.wdict[wx_str]
            mwx = self.wdict[awx_str]
            
            if c == "S11":
                sx = torch.mean(mwx**2, dim=(-1, -2))

                if derivatives:
                    wwx = self.wdict["IF((Fw)*(Fw)*(Fx))"]
                    ww = self.wdict["IF((Fw)*(Fw))"]

                    sx_grad = 4*torch.mean(wwx.real**2 * cov, dim=(-1, -2)) / self.Ntot
                    sx_hess = 2*ww[..., 0, 0].real * torch.mean(cov)

            elif c == "S00":
                sx = torch.mean(mwx**2, dim=(-1, -2)) - torch.mean(mwx, dim=(-1, -2))**2

                if derivatives:
                    ww = self.wdict["IF((Fw)*(Fw))"]
                    wwx = self.wdict["IF((Fw)*(Fw)*(Fx))"]
                    wsgwx = self.wdict[f"IF((Fw)*(F({sgwx_str})))"]
                    h1 = self.wdict[f"IF(F(w^2)*F(({ivmwx_str})*({sgwx_str}^2)))"]
                    h2 = self.wdict[f"IF(F(Aw^2)*F({ivmwx_str}))"]
                    mwx_mean = torch.mean(mwx, dim=(-1, -2), keepdim=True)

                    sx_grad = 4*torch.mean((wwx - mwx_mean*wsgwx).real ** 2 * cov, dim=(-1, -2)) / self.Ntot
                    sx_hess = torch.mean((2*ww[..., :1, :1] - 2/self.Ntot*wsgwx.real**2 + mwx_mean*(h1 - h2)).real * cov, dim=(-1, -2))

            elif c == "S01":
                sx = torch.mean(mwx*wx, dim=(-1, -2))

                if derivatives:
                    wmwx = self.wdict[f"IF((Fw)*(F({awx_str})))"]
                    g1 = self.wdict[f"IF((Fw)*(F({sgwx_str}*{wx_str})))"]
                    h1 = self.wdict[f"IF(F(Aw^2)*F({sgwx_str}))"]
                    h2 = self.wdict[f"IF(F(w^2)*F({sgwx_str}))"]
                    h3 = self.wdict[f"IF(F(w^2)*F({sgwx_str}^3))"]

                    sx_grad = torch.mean(torch.absolute(3*wmwx + torch.conj(g1))**2*cov, dim=(-1, -2)) / (4*self.Ntot)
                    sx_hess = torch.mean((6*torch.conj(h1) + 3*h2 - torch.conj(h3))*cov, dim=(-1, -2)) / 4

            elif c == "C01":
                mwx_c01 = mwx[..., self.wdict._c01_psi1_indices, :, :]
                wx_c01 = wx[..., self.wdict._c01_psi2_indices, :, :]
                sx = torch.mean(mwx_c01*wx_c01, dim=(-1, -2))

                if derivatives:
                    w1 = self.wdict["w"][self.wdict._c01_psi1_indices]
                    w2 = self.wdict["w"][self.wdict._c01_psi2_indices]
                    fw1 = self.wdict["Fw"][self.wdict._c01_psi1_indices]
                    fw21 = self.wdict["F(w^2)"][self.wdict._c01_psi1_indices]
                    fmw21 = self.wdict["F(Aw^2)"][self.wdict._c01_psi1_indices]
                    fw2 = self.wdict["Fw"][self.wdict._c01_psi2_indices]
                    fw1w2 = torch.fft.fft2(w1*w2)
                    fw1fw2c = torch.fft.fft2(w1*torch.conj(w2))

                    sgwx = self.wdict[sgwx_str]
                    fmwx = self.wdict[f"F({awx_str})"]
                    fmwx1 = fmwx[self.wdict._c01_psi1_indices]
                    mwx1 = mwx_c01
                    wx2 = wx_c01
                    sgwx1 = sgwx[self.wdict._c01_psi1_indices]
                    fsgwx1 = torch.fft.fft2(sgwx1)
                    sgcx = wx2 / mwx1
                    fsgcx = torch.fft.fft2(sgcx)

                    g1 = torch.fft.ifft2(fw1*torch.fft.fft2(sgwx1*torch.conj(wx2)))
                    g2 = torch.conj(torch.fft.ifft2(fw1*(torch.fft.fft2(sgwx1*wx2))))
                    g3 = torch.fft.ifft2(fw2*fmwx1)

                    h1 = torch.conj(torch.fft.ifft2(fmw21*fsgcx))
                    h2 = torch.fft.ifft2(fw1w2*fsgwx1)
                    h3 = torch.conj(torch.fft.ifft2(fw1fw2c*fsgwx1))
                    h4 = torch.fft.ifft2(fw21*torch.fft.fft2(torch.conj(sgcx)*sgwx1**2))
                    h5 = torch.conj(torch.fft.ifft2(fw21*torch.fft.fft2(sgcx*sgwx1**2)))

                    sx_grad = torch.mean(torch.absolute(g1 + g2 + 2*g3)**2*cov, dim=(-1, -2)) / (4*self.Ntot)
                    sx_hess = torch.mean((2*h1 + 4*h2 + 4*h3 - h4 - h5)*cov, dim=(-1, -2)) / 4

            stats_list.append(sx)
            if derivatives:
                stats_grad_list.append(sx_grad)
                stats_hess_list.append(sx_hess)
        sx_all = torch.cat(stats_list, dim=-1)
        if derivatives:
            sx_grad_all = torch.cat(stats_grad_list, dim=-1)
            sx_hess_all = torch.cat(stats_hess_list, dim=-1)

        self.wdict.del_var('x')

        if derivatives:
            return sx_all, sx_grad_all, sx_hess_all
        else:
            return sx_all
    
    def compute_s11_norm(self, x):
        self.wdict.add_var('x', x)
        stats_list = []
        for c in self.model:
            # Convenience strings
            awx_str = "A(IF((Fw)*(Fx)))"
            mwx = self.wdict[awx_str]
            
            if c in ["S11", "S00", "S01"]:
                sx = torch.mean(mwx**2, dim=(-1, -2))
            elif c == "C01":
                mwx_c01_1 = mwx[..., self.wdict._c01_psi1_indices, :, :]
                mwx_c01_2 = mwx[..., self.wdict._c01_psi2_indices, :, :]
                sx = torch.sqrt(torch.mean(mwx_c01_1**2, dim=(-1, -2)) * torch.mean(mwx_c01_2**2, dim=(-1, -2)))
            stats_list.append(sx)
        self.wdict.del_var('x')
        return torch.cat(stats_list, dim=-1)
