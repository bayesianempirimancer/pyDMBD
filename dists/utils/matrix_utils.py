import torch
class matrix_utils():

    def block_diag_matrix_builder(A,B):
        # builds a block matrix [[A,B],[C,D]] out of compatible tensors
        n1 = A.shape[-1]
        n2 = B.shape[-1]
        t_shape = A.shape[:-2]
        return torch.cat((torch.cat((A, torch.zeros(t_shape + (n1,n2),requires_grad=False)),-1),torch.cat((torch.zeros(t_shape + (n2,n1),requires_grad=False), B),-1)),-2)

    def block_matrix_inverse(A,B,C,D,block_form=True):
            # inverts a block matrix of the form [A B; C D] and returns the blocks [Ainv Binv; Cinv Dinv]
        invA = A.inverse()
        invD = D.inverse()
        Ainv = (A - B@invD@C).inverse()
        Dinv = (D - C@invA@B).inverse()
        
        if(block_form == 'left'):     # left decomposed returns abcd.inverse = [A 0; 0 D] @ [eye B; C eye]
            return Ainv, -B@invD, -C@invA, Dinv
        elif(block_form == 'right'):  # right decomposed returns abcd.inverse =  [eye B; C eye] @ [A 0; 0 D]
            return Ainv, -invA@B, -invD@C, Dinv            
        elif(block_form == 'True'):
            return Ainv, -Ainv@B@Dinv, -invD@C@invA, Dinv
        else:
            return torch.cat((torch.cat((Ainv, -invA@B@Dinv),-1),torch.cat((-invD@C@Ainv, Dinv),-1)),-2)

    def block_matrix_builder(A,B,C,D):
        # builds a block matrix [[A,B],[C,D]] out of compatible tensors
        return torch.cat((torch.cat((A, B),-1),torch.cat((C, D),-1)),-2)

    def block_precision_marginalizer(A,B,C,D):
        # When computing the precision of marginals, A - B@invD@C, does not need to be inverted
        # This is because (A - B@invD@C).inverse is the marginal covariance, the inverse of which is precsion
        # As a result in many applications we can save on computation by returning the inverse of Joint Precision 
        # in the form [A_prec 0; 0 D_prec] @ [eye B; C eye].  This is particularly useful when computing 
        # marginal invSigma and invSigmamu since    invSigma_A = A_prec
        #                                         invSigmamu_A = invSigmamu_J_A - B@invD@invSigmamu_J_B
        #                                           invSigma_D = D_prec
        #                                         invSigmamu_D = invSigmamu_J_D - C@invA@invSigmamu_J_A
         
        invA = A.inverse()
        invD = D.inverse()
        A_prec = (A - B@invD@C)
        D_prec = (D - C@invA@B)

        return A_prec, -B@invD, -C@invA, D_prec


    def block_matrix_logdet(A,B,C,D,singular=False):
        if(singular == 'A'):
            return D.logdet() + (A - B@D.inverse()@C).logdet()
        elif(singular == 'D'):            
            return A.logdet() + (D - C@A.inverse()@B).logdet()
        else:
            return D.logdet() + (A - B@D.inverse()@C).logdet()

