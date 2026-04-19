cv_mpliable<-function(fit,nfolds,X,Z,y,alpha,nlambda,new_t,maxgrid,tol,lambda_min,for_v,sv,fq,st,mv,ms,foldid=NULL,max_iter=1){
  BIG=10e9
  no<-nrow(X)
  ni<-ncol(X)
  nz<-ncol(Z)
  ggg=vector("list",nfolds)
  
 # yhat=array(NA,c(no,length(fit$lambda)))
  
  yhat=matrix(NA,nfolds,length(result$Lambdas))
  my_nzero<-matrix(0,nfolds,length(result$Lambdas))
  
  
  if(is.null(foldid)) foldid = sample(rep(1:nfolds, ceiling(no/nfolds)), no, replace=FALSE)  #foldid = sample(rep(seq(nfolds), length = no))
  
  nfolds=length(table(foldid))
  
  status.in=NULL
  
  for(ii in 1:nfolds){
    print(c("fold: ", ii))
    oo=foldid==ii
    
   
    
    ggg[[ii]]<-   plasso_fit1(y=y[!oo], X=X[!oo,,drop=F], Z=Z[!oo,,drop=F], nlambda=length(fit$Lambdas), alpha = alpha,new_t=new_t,my_mbeta=-.09,intercept = 0.05,step=.02,number=5,maxgrid=50,tol= tol,run=2,lambda_min=lambda_min,my_lambda=fit$Lambdas,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,cv_run=0,max_iter=max_iter)
    
    cv_p<-predict_lasso(ggg[[ii]] ,X=X[oo,,drop=F],Z=matrix(as.numeric(Z[oo,]),ni,nz),y=y[oo])
    
    
  #  Coeff<-lapply(seq_len(max(y)),
   #               function(j)(matrix(0,ncol(X),nlambda)))
    
    
    #for (i in 1:nlambda) {
    #  f<-Matrix(unlist(ggg[[ii]]$beta[i]),ncol(X),max(y),sparse = T)
     # for (j in 1:max(y)) {
      #  Coeff[[j]][,i]<-f[,j]
      #}
      
   # }
    
    #nnn_zero<-matrix(0,max(y),nlambda)
    
    #for (i in 1:max(y)) {
    #  q<-matrix(unlist(Coeff[[i]]),ncol(X),nlambda)
     # nnn_zero[i,]<-colSums(q!=0)
  
      
   # }
    #non_zero<-matrix(0,nlambda)
    #for (i in 1:nlambda) {
     # non_zero[i]<-max(nnn_zero[,i])
    #}
    
    #print(cv_p$deviance)
   # my_nzero[ii,]<-non_zero
    yhat[ii,]<- c(cv_p$deviance)
    #(result ,X,Z,y,lambda=NULL)
  }
  err=yhat
  non_zero<-matrix(0,nlambda)
  
  non_zero<-c(result$path$nzero)
  cvm=apply(yhat,2,mean,na.rm=T)
  #cvm.nz=cvm; cvm.nz[non_zero==0]=BIG
  #imin=which.min(cvm.nz)
  
  nn=apply(!is.na(err),2,sum,na.rm=T)
  #print(nn)
  cvsd=sqrt(apply(err,2,var,na.rm=T)/nn)
  cvm.nz=cvm; cvm.nz[non_zero==0]=BIG
  imin=which.min(cvm.nz)
  imin.1se=which(cvm< cvm[imin]+cvsd[imin])[1]
  
  
  out=list(lambda=(fit$Lambdas)[c(1:length(cvm))],cvm=cvm,cvsd=cvsd,cvup = cvm +
             cvsd, cvlo = cvm - cvsd, nz=c(result$path$nzero)[c(1:length(cvm))],lambda.min=fit$Lambdas[imin],lambda.1se=fit$Lambdas[imin.1se])
  
 
  return(out) 
}




quad_solution<-function(u, v, w){
  temp = ((v^2) - (4 * u * w))^0.5
  root1 = (-v + temp) / (2 * u)
  root2 = (-v - temp) / (2 * u)
  roots<-list(root1, root2)
  return (roots)
}

##############
####### iterations 
quadratic=function( beta ,theta, alpha,lambda,beta0,theta0,j,b,W,X,Z,y,N,n_i,xbar,zbar,big_delta_1=NULL,big_delta_2=NULL,b_1=NULL,b_2=NULL,t=NULL,r_min_j){
  #print(c(L1,L2))

  if(isTRUE(is.null(big_delta_1))==T ){big_delta_1=0; lambda_d=0}else{big_delta_1=big_delta_1;lambda_d=lambda}
  if(isTRUE(is.null(big_delta_2))==T ){big_delta_2=0}else{big_delta_2=big_delta_2}
  # print(big_delta)
  
  if(isTRUE(is.null(b_1))==T){b_1=matrix(rep(0,p))}else{
    
    b_1=matrix(b_1)
  }
  
  # print(b_1)
  if(isTRUE(is.null(b_2))==T){b_2=matrix(0,p,1)}else{
    
    b_2=b_2
  }
  
  
  
	U<-matrix(0); U2<-matrix(0); U3<-matrix(0)
#N=(nrow(X))
	

#t=1;nes=1; beta_u<-matrix(beta,p,1) ; theta_u<- matrix(theta,p,K) 
#Beta_new<-matrix(beta,p,1); Theta_new<- matrix(theta,p,K); v_beta=matrix(beta,p,1);v_theta<- matrix(theta,p,K)
# U<-matrix(0); U2<-matrix(0); U3<-matrix(0)
# N=(nrow(X))

	big=10e9; eps = 1e-5
#print(beta_new)

t=1;nes=1; beta_u=beta;theta_u=theta;beta_new=beta;theta_new=theta;v_beta=beta;v_theta=theta;n_l=n_i;n_r=n_i
j=j
# U<-matrix(0); U2<-matrix(0); U3<-matrix(0)
# N=(nrow(X))

okay=0


# 
# old_beta_u=beta_u; old_theta_u=theta_u
# teta_k<-2/(nes+1)
# 
# beta_new<-(1-teta_k)*old_beta_u+teta_k*v_beta
# theta_new<-(1-teta_k)*old_theta_u+teta_k*v_theta






while (okay<1) {
  #print(beta[j])
  # 
  # old_beta_u=beta_u; old_theta_u=theta_u
  # teta_k<-2/(nes+1)
  # 
  # beta_new<-(1-teta_k)*old_beta_u+teta_k*v_beta
  # theta_new<-(1-teta_k)*old_theta_u+teta_k*v_theta
  
  
  theta_transpose_l<-t(theta_u);theta_transpose_r<-t(theta_new)
  
  
  
  

  n_r[b] <- list(model(beta0, theta0, beta_new, theta_new, X, Z))
  
  
  E_r<-matrix(0,N,max(y))
  for (x in 1:max(y)) {
   
    E_r[,x]<-exp(unlist(n_r[x]))
  }
  
  n_r_b<-matrix(unlist(n_r[b]),N)
   y_hat_r=n_r_b
  
  
  pr_r<-(matrix(as.numeric(exp(n_r_b)/( (rowSums(E_r)))),N))
  
  	B_r<-pr_r*(1-pr_r)
  	B_r[B_r==0]=as.numeric(10e-9)
  
  y1=1*(y==b)
  
  
  M_r<-(y1-pr_r)
  
  A_r<-y_hat_r+M_r/B_r
  
  r_r<-B_r*(A_r-y_hat_r)
  
  
  
  
  
  
  grad_j<- gradient_j(beta=beta_new,theta=theta_new,U,U2,U3,y,X,W,r=r_r,alpha,lambda,K,j,N)
  
  
  L1=matrix(unlist(grad_j[1]));L2<-matrix(unlist(grad_j[2]))
  
  U=c(unlist(grad_j[3]));U2=c(unlist(grad_j[4]));U3=c(unlist(grad_j[5]))
  
  
  
  
  
  
  beta_j=as.numeric(beta_new[j]);theta_j=as.numeric(theta_new[j,])
  
  a<- as.numeric(norm(matrix(beta_j), type="F")) ; bb<-as.numeric(norm(matrix(theta_j), type="F")) ; rho_2<- as.numeric(sqrt(a^2+bb^2))
  
  
  ##### after norming 
  c<- as.numeric(t*(1-alpha)*lambda );  g_1<- as.numeric( abs(beta_j-t*as.numeric(L1)))
  v<-matrix(0,nrow = 1,ncol = K)
  
  # for (b in 1:K) {
  v<-S_func(theta_j-t*L2,t*alpha*lambda)
  # }
  
  g_2<-as.numeric(norm(matrix(v),type="F"))
  
  root = quad_solution(1, 2 * c, 2 * c * g_2 - g_1^ 2 - g_2^ 2)
  root1 <- unlist(root[1])
  root2<- unlist(root[2])
  # posroot <- -c+sqrt(c^2-2*c*g_2+g_1^2+g_2^2)
  # negroot<- -c-sqrt(c^2-2*c*g_2+g_1^2+g_2^2)
  
  # a: norm of beta, b: norm of theta
  # Hence, we choose the largest value of a and b to take positive value
  
  a = c(
    g_1 * root1 / (c + root1),
    g_1 * root2 / (c + root2),
    g_1 * root1 / (c + root2),
    g_1 * root2 / (c + root1)
  )
  
  bb = c(
    g_1*root1 * (c - g_2) / (c + root1),
    g_1*root2 * (c - g_2) / (c + root2),
    g_1*root1 * (c - g_2) / (c + root2),
    g_1*root2 * (c - g_2) / (c + root1)
  )
  
  
  
  
  x_min = big
  
  j_hat=0; k_hat = 0
  for (jjj in 1:4){
    for (kkk in 1:4){
      denominator = (a[jjj]^2 + bb[kkk]^ 2)^ 0.5  # l2 norm
      if (isTRUE(denominator > 0)==T){
        val1 = (1 + (c / denominator)) * a[jjj] - g_1
        val2 = (1 + c * (1 / bb[kkk] + 1 / denominator)) * bb[kkk] - g_2
        
        temp = abs(val1) + abs(val2)  # l1 norm
        if (isTRUE(temp < x_min)==T){
          j_hat=jjj; k_hat = kkk
          x_min = temp
        }
      }
      
      
      
    }
    
  }  
  
  xnorm = (a[j_hat]^2 + bb[k_hat]^2)^0.5  # l2 norm
  
  
  
  
  #beta_hat<-((beta_j-h*L1))/c_1
  #new_beta_hat<- beta_hat+ my_b*(beta_hat-old_beta_hat)
  new_v_beta<-((beta_j-t*L1))/(1 + c / xnorm)
  #print(new_v_beta)
  
  new_v_theta<- (matrix(S_func(theta_j-t*L2,t*alpha*lambda)))/(1 + c * ( (1 / xnorm) + (1 /abs(bb[k_hat])) ) )
  
  
  beta_u[j]<-new_v_beta;theta_u[j,]<-new_v_theta
  
  theta_transpose_l<-t(theta_u);theta_transpose_r<-t(theta_new)
  
 
  
  
  
  #my_X<-scale(X,-xbar,FALSE) ; my_Z<-scale(Z,-zbar,FALSE)
  #my_X=X ; my_Z=Z
  #my_X=matrix(as.numeric(my_X),N,p)
 # my_Z=matrix(as.numeric(my_Z),N,K)
  
 
  n_l[b] <- list(model(beta0, theta0, beta_u, theta_u, X, Z))
  
  
  
  n_r[b] <- list(model(beta0, theta0, beta_new, theta_new, X, Z))
  
  E_l<-matrix(0,N,max(y))
  E_r<-matrix(0,N,max(y))
  for (x in 1:max(y)) {
    E_l[,x]<-exp(unlist(n_l[x]))
    E_r[,x]<-exp(unlist(n_r[x]))
  }
  n_l_b<-matrix(unlist(n_l[b]),N)
  n_r_b<-matrix(unlist(n_r[b]),N)
  y_hat_l=n_l_b; y_hat_r=n_r_b
  
  
  pr_l<-(matrix(as.numeric(exp(n_l_b)/( (rowSums(E_l)))),N)); pr_r<-(matrix(as.numeric(exp(n_r_b)/( (rowSums(E_r)))),N))
  
  B_l<-pr_l*(1-pr_l);	B_r<-pr_r*(1-pr_r)
  
  
  y1=1*(y==b)
  
  
  M_l<-(y1-pr_l);M_r<-(y1-pr_r)
  
  A_l<-y_hat_l+M_l/B_l;A_r<-y_hat_r+M_r/B_r
  
  r_l<-B_l*(A_l-y_hat_l);r_r<-B_r*(A_r-y_hat_r)
  
  
  
  
  
  #l_penalty1<-sum(cvxr_norm(hstack(beta_u, theta_u), 2, axis = 1))
  
  
 
  
  #	for (g in 1:p) {
  
  #		pen1_l<-norm(matrix(c(beta_u[g],theta_u[g,])),type = "F");pen2_l<-norm(matrix(c(theta_u[g,])),type = "F")
  #		pen3_l<-sum(abs(theta_u[g,]))
  #		norm_1_l[g]<-lambda*(1-alpha)*(pen1_l+pen2_l)+lambda*alpha*pen3_l
  
  
 # norm_1_l=   lapply(seq_len(p),
 #                    function(g)(  lambda*(1-alpha)*(norm(matrix(c(beta_u[g],theta_u[g,])),type = "F") +norm(matrix(c(theta_u[g,])),type = "F") ) + lambda*alpha* sum(abs(theta_u[g,]))      ))
  
  # pen1_l<-norm(matrix(c(beta_u[g],theta_u[g,])),type = "F");pen2_l<-norm(matrix(c(theta_u[g,])),type = "F")
  # pen3_l<-sum(abs(theta_u[g,]))
  #  norm_1_l[g]<-lambda*(1-alpha)*(pen1_l+pen2_l)+lambda*alpha*pen3_l
  
 # norm_1_r= lapply(seq_len(p),
  #                 function(g)(  lambda*(1-alpha)*(norm(matrix(c(beta_new[g],theta_new[g,])),type = "F") +norm(matrix(c(theta_new[g,])),type = "F") ) + lambda*alpha* sum(abs(theta_new[g,]))      ))
  
  
  # 		pen1_r<-norm(matrix(c(beta_new[g],theta_new[g,])),type = "F");pen2_r<-norm(matrix(c(theta_new[g,])),type = "F")
  #		pen3_r<-sum(abs(theta_new[g,]))
  #		norm_1_r[g]<-lambda*(1-alpha)*(pen1_r+pen2_r)+lambda*alpha*pen3_r
  
  #	}
  
  
  
  
  #	print(c(l_penalty1,l_penalty2,l_penalty3))
  

  
  objective_l <- objective_j(beta0,theta0,beta,theta,X,Z,y,W,alpha,lambda,p,K,N,j,r_min_j,b,n_l,E_l,B=B_l)
  objective_r <- objective_j(beta0,theta0,beta_new,theta_new,X,Z,y,W,alpha,lambda,p,K,N,j,r_min_j,b,n_r,E_r,B=B_r)
  
  #objective_l <- sum(r_l*r_l) / (2 * N) +sum(unlist(norm_1_l))#+(sign(big_delta_1-lambda_d)/(2*N))*(norm(matrix(c(beta_u),ncol = 1),"F"))^2+(sign(big_delta_2-lambda_d)/(2*N))*(norm(matrix(c(theta_u),ncol = 1),"F"))^2 +(t(b_1)%*%beta_u)/N+sum(t(b_2)%*%theta_u)/N
  
  
  #objective_r <- sum(r_r*r_r) / (2 * N) +sum(unlist(norm_1_r))#+(sign(big_delta_1-lambda_d)/(2*N))*(norm(matrix(c(beta_new),ncol = 1),"F"))^2+(sign(big_delta_2-lambda_d)/(2*N))*(norm(matrix(c(theta_new),ncol = 1),"F"))^2+(t(b_1)%*%beta_new)/N+sum(t(b_2)%*%theta_new)/N
  
  
  rhs<-objective_r+ matrix(c(L1,L2),nrow = 1)%*%matrix((c(beta_u[j],theta_u[j,])-c(beta_new[j],theta_new[j,])),ncol = 1)+(1/(2*t))*(norm(matrix((c(beta_u[j],theta_u[j,])-c(beta_new[j],theta_new[j,])),ncol = 1),"F"))^2
  
  
  
  
  
  
  # 
  # if(isTRUE(objective_l<=rhs)==T & abs(beta_new[j])>= beta[j]){
  #   
  #   beta_new[j]<-beta_u[j]
  #   print(c(b,j,beta_new[j],beta[j],t,nes))
  #   #print(c(b,j,objective_l,rhs,t,nes))
  #   #beta_new[j]<-new_v_beta;theta_new[j,]<-new_v_theta
  #   okay=1
  # }else 
  if(isTRUE(objective_l<=rhs)==T ){
    # if(abs(beta_u[j])<abs(beta[j])  ){
    #   beta_u[j]=beta[j]
    # }
     # print(c(b,j,beta_u[j],beta[j],t,nes))
      #print(c(b,j,objective_l,rhs,t,nes))
      #beta_new[j]<-new_v_beta;theta_new[j,]<-new_v_theta
      okay=1
    }  else{
    
      old_beta=v_beta;old_theta=v_theta
      v_beta=beta_u
      v_theta=theta_u
      beta_new=v_beta+(nes/(nes+3))*(v_beta-old_beta)
      theta_new=v_theta+(nes/(nes+3))*(v_theta-old_theta)
      
   
   # print(beta_new)
    #print(theta_new)
    nes=nes+1
    t=.8*t
  }
  
  if(isTRUE(nes>200)==T){
    #beta_u[j]=beta[j]
   # print(c(b,j,beta_u[j],beta[j],t,nes))
    okay=1
  }
  
}#while

beta1<-beta_u[j];theta1<-theta_u[j,];t=t

#while

 # print(c(b,j))
#print(c(b,j))
#print(beta1)
#print(theta1)
	return(list(beta1,theta1,t))
}
