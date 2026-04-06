#TRIAL 1
library(glmnet)



N = 100 ; p =10;nz=4; K=nz
#X <- matrix(rnorm(n = N * p, mean = 0, sd = 1), nrow = N, ncol = p)
#sim1_X<-X
#Z <- matrix(rbinom(n = N * K, size = 1, prob = 0.5), nrow = N, ncol = K)
#sim1_Z<-Z
X <- matrix(rnorm( N * p), nrow = N, ncol = p)

mx=colMeans(X)

sx=sqrt(apply(X,2,var))
X=scale(X,mx,sx)

X=matrix(as.numeric(X),N,p)

#Z <- matrix(rbinom(n = N * K, size = 1, prob = 0.5), nrow = N, ncol = K)
Z =matrix(rnorm(N*nz),N,nz)
mz=colMeans(Z)
sz=sqrt(apply(Z,2,var))

Z=scale(Z,mz,sz)
e=matrix(1,N)
beta_1 <- rep(x = 0, times = p); beta_2<-rep(x = 0, times = p);beta_3<-rep(x = 0, times = p)
beta_1[1:5] <- c(2, 2, 2, 2,2); beta_2[1:5]<-c(-2,  -2,-2, -2,-2); beta_3[1:5]<-c(4, -2, 1, 3,2)
#beta_1[1:5]<-c(4,-2,2,2,-5)
coeffs1 <- cbind(beta_1[1]+5*Z[,1], beta_1[2], beta_1[3] +  3*Z[, 2],  beta_1[4] *(e -  2*Z[, 3]),beta_1[5]*(e-2*Z[,4]))+.5*rnorm(N)
coeffs2 <- cbind(beta_2[1]+5*Z[,1], beta_2[2], beta_2[3] +  3*Z[, 2],  beta_2[4] *(e -  2*Z[, 3]),beta_2[5]*(e-2*Z[,4]))+.5*rnorm(N)
coeffs3 <- cbind(beta_3[1]+5*Z[,1], beta_3[2], beta_3[3] +  3*Z[, 2],  beta_3[4] *(e -  2*Z[, 3]),beta_3[5]*(e-2*Z[,4]))+.5*rnorm(N)





#c1=matrix(0,N,p);c2=matrix(0,N,p);c3=matrix(0,N,p)
#c1[,c(1:5)]<-coeffs1;c2[,c(1:5)]<-coeffs2;c3[,c(1:5)]<-coeffs3

# 
# coeffs1 <- cbind(beta_1[1],beta_1[2], beta_1[3] -  3*Z[, 1]+3*Z[,2],  beta_1[4]  -  2*Z[, 3]+2*Z[,4],beta_1[5])
#coeffs2 <- cbind(beta_2[1],beta_2[2], beta_2[3] -  3*Z[, 1]+3*Z[,2],  beta_2[4]  -  2*Z[, 3]+2*Z[,4],beta_2[5])
# 
# coeffs3 <- cbind(beta_3[1],beta_3[2], beta_3[3] -  3*Z[, 1]+3*Z[,2],  beta_3[4]  -  2*Z[, 3]+2*Z[,4],beta_3[5])



vProb = cbind(exp(diag(X[, 1:5]%*%t(coeffs1))), exp(diag(X[, 1:5]%*%t(coeffs2))), exp(diag(X[, 1:5]%*%t(coeffs3))))
mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
y = apply(mChoices, 1, function(x) which(x==1))
num_eff=5
#y_1<-4*X[,1] -2*X[,2]+2*X[,3] +2*X[,4]-5*X[,5]-2*X[,1]*Z[,1]+-2*X[,1]*Z[,2]+X[,2]*(4*Z[,3]-Z[,4])+ 4*rnorm(N)

#y1=1*(y_1>=0 & y_1<=3)
#y2=2*(y_1<0)
#y3=3*(y_1>3)
#e=matrix(y1+y2+y3)
#Z=matrix(as.numeric(Z),N,nz)
#y=e
table(y)
#dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), X)
#y=gendata_MLR(N, p, NC = 3, X = X, betas = coeffs1)


#mu <- 4*X[,1] -2*X[,2]+X[,3] +5*X[,1]*Z[,1]+3*X[,3]*Z[,2]+X[,4]*(3-2*Z[,3])+ 3*rnorm(N )  #diag(X[, 1:5] %*% t(coeffs))
#y <- mu + 4 * rnorm(N, mean = 0, sd = 1)


#y =mu + 0.5 * rnorm(N, mean = 0, sd = 1)

#y1=1*(y>= -1 & y<=3)


#y2=2*(y< -1)

#y3=3*(y>3)


#e=matrix(y1+y2+y3)


#Z=matrix(as.numeric(Z),N,nz)
#y=e
#table(y)


nlambda=50

#system.time(result <- plasso_fit1(y=y, X, Z, nlambda, alpha = .5,new_t=1.,my_mbeta=-.02,intercept = 0.001,step=-.05,number=0,maxgrid=50,tol= -1.4e-1,run=2))
tt=rep(t,nlambda)
for_v=10;sv=0;fq=50;st=50;mv=20;ms=50;tol=1e-3
system.time(result <- plasso_fit1(y=y, X, Z, nlambda, alpha = .5,new_t=1,my_mbeta=.09,intercept = 0.01,step=.05,number=10,maxgrid=50,tol= tol,run=2,lambda_min = .001,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,cv_run = 0,max_iter=10000))



#p_lasso=pliableMulti(y=y,X,Z,nlambda = nlambda,alpha=.5,family = "multinomial")

path<-result$path

path


big_t<-result$big_T
Mbeta=result$Mbeta
Lambda<-as.matrix(result$Lambdas)
glmnet.control(fdev = 0)
# 
#fit1<-glmnet(x=X,y=y,alpha = 1,family = "multinomial" ,nlambda =50)
# 
# fit2<-glmnet(x=X,y=y,alpha = 0.5,family = "multinomial" ,nlambda = 50)
#data_gbm<-data.frame(Y=y,X)
#fit3$estimator

dat <- data.frame( X, Z)
names(dat) <- c(paste0("X", 1:ncol(X)),paste0("Z", 1:ncol(Z)) )
#f <- as.formula(y ~ .*.)
#f <- as.formula(y~ (X1 +X1:Z1)+X2+(X3+X3:Z2)+(X4-X4:Z3)+(X5-X5:Z4)+X6+X7+X8+X9+X10  +0)
#cbind(beta_1[1]+5*Z[,1], beta_1[2], beta_1[3] +  3*Z[, 2],  beta_1[4] *(e -  2*Z[, 3]),beta_1[5]*(e-2*Z[,4]))+.5*rnorm(N)

y_Z <- y
X_Z <- as.matrix(dat)#model.matrix(f, data=dat)[, -1]


fit3<-glmnet(x=X_Z,y=y_Z,alpha = 1,family = "multinomial" ,nlambda =50)

fit4<-glmnet(x=X_Z,y=y_Z,alpha = 0.5,family = "multinomial" ,nlambda = 50)


fitdiv1<-deviance(fit3)/length(y)
fitdiv2<-deviance(fit4)/length(y)


plot(log(fit3$lambda),fitdiv1, type="o", col="red",xlab = "log(lambda)", ylab = "Train Deviance",ylim =c(min((c(path$DEV,fitdiv1,fitdiv2))), max((c(path$DEV,fitdiv1,fitdiv2)))),xlim =c(min(c(log(as.matrix(Lambda)),log(fit3$lambda),log(fit4$lambda))),max(c(log(as.matrix(Lambda)),log(fit3$lambda),log(fit4$lambda)))))
lines(log(Lambda)[c(1:50)],path$DEV[c(1:50)], col="blue",type = "o")
#lines(log(fit2$lambda)[c(1:50)],fit2$dev.ratio[c(1:50)], type="o", col="green")
#lines(log(fit3$lambda)[c(1:50)],fit3$dev.ratio[c(1:50)], type="o", col="yellow")
lines(log(fit4$lambda)[c(1:50)],fitdiv2[c(1:50)], type="o", col="green")


legend(-10, 2, legend=c("lasso", "plasso","elastic net"),
       col=c("red", "blue","green"), lty=1:2, cex=0.8)


data<-data.frame(M_pliable=path$DEV[c(1:50)],lasso=fitdiv1,elastic_net=fitdiv2[c(1:50)])
boxplot(data, main="N=100, p=10, K=4", ylab="Train Deviance")

plot(log(Lambda),path$DEV,type="o", col="red",xlab = "log(lambda)", ylab = " Deviance")







# 
# 
# Coeff<-lapply(seq_len(max(y)),
#               function(j)(matrix(0,ncol(X),nlambda)))
# 
# 
# for (i in 1:nlambda) {
#   f<-matrix(unlist(result$beta[i]),ncol(X),max(y))
#   for (j in 1:max(y)) {
#     Coeff[[j]][,i]<-f[,j]
#   }
#   
# }
# 



for (r in 1:max(y)) {
  # q<-matrix(unlist(Coeff[[i]]),ncol(X),nlambda)
  
  gg=nlambda
  my_beta<-matrix(0,p,gg)
  
  
  #result$beta[[r]][1,]
  for (i in 1:p) {
    
    
    my_beta[i,]<-result$beta[[r]][i,]
    
    
  }
  
  
  b=apply(abs(my_beta), 2, sum)
  b=c(1:gg)
  n=dim(my_beta)[2]
  matplot(b,cbind( t(my_beta),fit$path$OBJ_main[c(1:gg)]),type="n",col="red", ylim = range(my_beta),  xlab="nlambda", ylab= ( paste(  "Coefficients", r)) )
  
  #plot(b,result$beta[[r]][1,],type="l",col="red", ylim = range(result$beta[[r]]), ylab=paste(  "Coefficients", r)   ,xlab = "nlambda")
  axis(side = 3, at =  (as.matrix(b)), labels = paste(as.matrix(result$non_zero[,r])),
       tick = FALSE, line = 0)
  
  for (i in 1:num_eff) {
    lines(b,(my_beta[i,]),col=i+1, lty=1  )
    
    text( (b[n]+1),my_beta[i,n], labels = i)
  }
  for (i in (num_eff+1):p) {
    lines(b,(my_beta[i,]),col=i+1, lty=2  )
    
    text( (b[n]+1),my_beta[i,n], labels = i)
  }
  
  
  
  
  act=which(rowSums(abs(my_beta))>0)
  theta1<-array(0,c(ncol(X),ncol(Z),(nlambda) ))
  for (i in 1:(nlambda)) {
    theta1[,,i]<-matrix(unlist(result$theta[[r]][,,i]),p,K)
  }
  
  
  ntheta=  apply( abs(theta1)>0,c(1,3),sum)
  index = c(1:gg)#apply(abs(my_beta), 2, sum)+apply(abs(theta1),3,sum)
  sbeta=(my_beta)
  for(j in act){
    for(i in 1:length(index)){
      
      if(ntheta[j,i]>0) text(index[i],sbeta[j,i],label="x",cex=.7)
    }}
  
  
  
  
  
}

#plot(fit1,label=T)

#plot(fit2,label=T)

plot(fit3,label=T)

plot(fit4,label=T)


####cv for trial 1
####### Z trained########
nfolds=5
set.seed(3321)
foldid = sample(rep(seq(nfolds), length = N))
foldid = sample(rep(1:nfolds, ceiling(N/nfolds)), N, replace=FALSE)
# predict z  from x; here we use glmnet, but any other supervised learning procedure
#  could be used
zhat=matrix(NA,N,ncol(Z))
for(ii in 1:nfolds){
  zfit=cv.glmnet(X[foldid!=ii,],Z[foldid!=ii,],family="mgaussian")
  zhat[foldid==ii,]=predict(zfit,X[foldid==ii,],s=zfit$lambda.min)
}

#NOTE that the same foldid vector must be passed to cv.pliable
#this is used when you want to predict Z from X
#cvfit=cv.pliable(fit,x,zhat,y,foldid=foldid)
#plot(cvfit)
####### ###################
system.time(result_cv <- cv_mpliable(fit=result,nfolds=5,X,Z=zhat,y,alpha=.5,new_t=.5,nlambda=50,maxgrid=50,tol=tol,lambda_min=0.0001,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,foldid=foldid),max_iter=1000)


result_cv


#result_cv. This used when you want to generate random values for Z
###################################################

system.time(result_cv <- cv_mpliable(fit=result,nfolds=5,X,Z,y,alpha=.5,new_t=.5,nlambda=50,maxgrid=50,tol=tol,lambda_min=0.0001,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,max_iter=1000))
#system.time(result_cv <- cv_mpliable(fit=result,nfolds=5,X,Z,y,alpha=.5,nlambda=50,maxgrid=50,for_v=for_v,sv=sv,fq=fq,st=st))


result_cv

#result_cv




cvfit1<-cv.glmnet(x=X_Z,y=y_Z,alpha = 1,family = "multinomial", nfolds = 5,nlambda=nlambda)


cvfit2<-cv.glmnet(x=X_Z,y=y_Z,alpha = .5,family = "multinomial", nfolds = 5,nlambda=nlambda)
#deviance from (test samples? all samples?) #at lambda value


plot(cvfit1)
plot(cvfit2)





plot(log(as.matrix(result_cv$lambda)),(as.matrix(result_cv$cvm)), type = 'o', xlab = "log(lambda)",ylab = "Deviance",pch = 20,col="red",ylim=c(min(c(result_cv$cvm,(result_cv$cvup)[c(1:length(result_cv$cvm))],result_cv$cvlo[c(1:length(result_cv$cvm))] )),max(c(result_cv$cvm,(result_cv$cvup)[c(1:length(result_cv$cvm))],result_cv$cvlo[c(1:length(result_cv$cvm))])))  )

abline(v= log(result_cv$lambda.min), lty = 3)
#abline(v= log(plasso_cv$lambda.min), lty = 3)
abline(v= log(result_cv$lambda.1se), lty = 3)
axis(side = 3, at =  log(as.matrix(result_cv$lambda)), labels = paste(as.matrix(result_cv$nz)),
     tick = FALSE, line = 0)


error.bars( log(result_cv$lambda), result_cv$cvup, result_cv$cvlo,
            width = 0.01, col = "darkgrey")





lambda<-as.numeric(result_cv$lambda.min)


#z1<-1*(Z[,1]==1);z2<-2*(Z[,2]==1);z3<-3*(Z[,3]==1);z4<-4*(Z[,4]==1)
#my_Z<-z1+z2+z3+z4
#to predict Z form X
#note: if you don't use this , then Z_test must be genrated same way as X_test
zfit<-cv.glmnet(X,zhat,family="mgaussian")

######################3#######################3##############3#
######################PREDICT#####################3##########

N_test = 500 ; p =p;nz=nz; K=nz
#X_test <- matrix(rnorm(n = N_test * p, mean = 0, sd = 1), nrow = N_test, ncol = p)
X_test <- matrix(rnorm( N_test * p), nrow = N_test, ncol = p)

#mx=colMeans(X_test)

#sx=sqrt(apply(X_test,2,var))
X_test=scale(X_test,mx,sx)
#X_test=matrix(as.numeric(X_test),N_test,p)

#to predict Z_test
Z_test =predict(zfit,X_test,s=zfit$lambda.min)[,,1]


# to generate random Z
# 
Z_test =matrix(rnorm(N_test*nz),N_test,nz)
mz=colMeans(Z_test)
sz=sqrt(apply(Z_test,2,var))

Z_test=scale(Z_test,mz,sz)
#Z_test <- matrix(rbinom(n = N_test * K, size = 1, prob = 0.5), nrow = N_test, ncol = K)
#Z_test=zhat
e=matrix(1,N_test)
beta_1 <- rep(x = 0, times = p); beta_2<-rep(x = 0, times = p);beta_3<-rep(x = 0, times = p)
beta_1[1:5] <- c(2, 2, 2, 2,2); beta_2[1:5]<-c(-2,  -2,-2, -2,-2); beta_3[1:5]<-c(4, -2, 1, 3,2)
# beta_1[1:5]<-c(4,-2,2,2,-5)
coeffs1 <- cbind(beta_1[1]+5*Z_test[,1], beta_1[2], beta_1[3] +  3*Z_test[, 2],  beta_1[4] *(e -  2*Z_test[, 3]),beta_1[5]*(e-2*Z_test[,4]))+.5*rnorm(N)
coeffs2 <- cbind(beta_2[1]+5*Z_test[,1], beta_2[2], beta_2[3] +  3*Z_test[, 2],  beta_2[4] *(e -  2*Z_test[, 3]),beta_2[5]*(e-2*Z_test[,4]))+.5*rnorm(N)
coeffs3 <- cbind(beta_3[1]+5*Z_test[,1], beta_3[2], beta_3[3] +  3*Z_test[, 2],  beta_3[4] *(e -  2*Z_test[, 3]),beta_3[5]*(e-2*Z_test[,4]))+.5*rnorm(N)





# 
# coeffs1 <- cbind(beta_1[1], beta_1[2], beta_1[3] -  3*Z_test[, 1]+3*Z_test[, 2],  beta_1[4]  -  2*Z_test[, 3]+2*Z_test[,4],beta_1[5])
# coeffs2 <-cbind(beta_2[1], beta_2[2], beta_2[3] -  3*Z_test[, 1]+3*Z_test[, 2],  beta_2[4]  -  2*Z_test[, 3]+2*Z_test[,4],beta_2[5])
# coeffs3 <- cbind(beta_3[1], beta_3[2], beta_3[3] -  3*Z_test[, 1]+3*Z_test[, 2],  beta_3[4]  -  2*Z_test[, 3]+2*Z_test[,4],beta_3[5])

vProb = cbind(exp(diag(X_test[, 1:5]%*%t(coeffs1))), exp(diag(X_test[, 1:5]%*%t(coeffs2))), exp(diag(X_test[, 1:5]%*%t(coeffs3))))
mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
y_test = apply(mChoices, 1, function(x) which(x==1))






table(y_test)





#Z_test=matrix(as.numeric(Z_test),N_test,nz)




dat_test <- data.frame( X_test, Z_test)
names(dat_test) <- c(paste0("X", 1:ncol(X_test)),paste0("Z", 1:ncol(Z_test)))
#f <- as.formula(y~X1 +X1:Z1+X2+X3+X3:Z2+X4-X4:Z3+X5-X5:Z4  +0)
y_Z_test <- y_test
#X_Z_test<- model.matrix(f, dat_test)[, -1]

X_Z_test<-as.matrix(dat_test)



to_run=50

dev_P<-matrix(0,to_run); dev_L<-matrix(0,to_run);dev_E<-matrix(0,to_run)
for (n in 1:to_run) {
  
  
  predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=Lambda[n])
  dev_P[n]<-predict$deviance
  
  glmnet_pred1<-predict(fit3,newx=X_Z_test,type="response",s=cvfit1$lambda[n])
  pr_l<- data.frame(glmnet_pred1)
  glmnet_pred2<-predict(fit4,newx=X_Z_test,type="response",s=cvfit2$lambda[n])
  pr_e<- data.frame(glmnet_pred2)
  
  Dev_l<-matrix(0,nrow = length(y_test))
  Dev_e<-matrix(0,nrow = length(y_test))
  for (l in 1:length(Dev_l)) {
    deviance_y1<-matrix(0,1,max(y_test))
    deviance_y2<-matrix(0,1,max(y_test))
    for (d in 1:max(y_test)) {
      y_d<-1*(y_test[l]==d)
      # n_i_d<-matrix(unlist(n_i[d]),N)
      prob_d_l <-matrix(unlist(pr_l[,d]),N_test)
      prob_d_e <-matrix(unlist(pr_e[,d]),N_test)
      # deviance_y1[d]<-y_d*n_i_d[l]
      # deviance_y2[d]<-exp(n_i_d[l])
      deviance_y1[d]<-  Log(y=y_d,pr=prob_d_l[l])
      deviance_y2[d]<-  Log(y=y_d,pr=prob_d_e[l]) 
      #+(1-y_d)*Log(y=(1-y_d),pr=(1-prob_d[l]))
    }
    Dev_l[l]<-  sum(deviance_y1)
    Dev_e[l]<-  sum(deviance_y2)
  }
  
  # Dev_l<-matrix(0,nrow = length(y_test))
  # Dev_e<-matrix(0,nrow = length(y_test))
  # for (l in 1:length(Dev_l)) {
  #   deviance_y1<-matrix(0,1,max(y_test))
  #   deviance_y2<-matrix(0,1,max(y_test))
  #   for (d in 1:max(y_test)) {
  #     y_d<-1*(y_test[l]==d)
  #     # n_i_d<-matrix(unlist(n_i[d]),N)
  #     prob_d_l <-matrix(unlist(pr_l[,d]),N_test)
  #     prob_d_e <-matrix(unlist(pr_e[,d]),N_test)
  #     # deviance_y1[d]<-y_d*n_i_d[l]
  #     # deviance_y2[d]<-exp(n_i_d[l])
  #     deviance_y1[d]<-sum(errfun.binomial(y=y_d,yhat=prob_d_l[l],w=rep(1,length(y_d)))) #+(1-y_d)*Log(y=(1-y_d),pr=(1-prob_d[l]))
  #     #print(deviance_y1[d])
  #     deviance_y2[d]=sum(errfun.binomial(y=y_d,yhat=prob_d_e[l],w=rep(1,length(y_d))))
  #   }
  #   Dev_l[l]<-  sum(deviance_y1)
  #   Dev_e[l]<-  sum(deviance_y2)
  #   
  #   
  #   #(y,yhat,w=rep(1,length(y)))
  # }
  dev_L[n]<-((2)*sum(Dev_l))/length(y_test)
  dev_E[n]<-((2)*sum(Dev_e))/length(y_test)
  
  
  
  #dev_L[n]<- ((2)*sum(Dev_l))#
  #dev_E[n]<- ((2)*sum(Dev_e))#
}

test_deviance<- data.frame(plasso=dev_P,lasso=dev_L,E_net=dev_E)
boxplot(test_deviance,ylab="Test deviance")
##########################################################
plot(log(as.matrix(Lambda)[c(1:to_run)]),dev_P,type="o", col="red",ylim=c(min(c(dev_P,dev_L,dev_E)),max(c(dev_P,dev_L,dev_E))),xlim =c(min(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))),max(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))) ) ,xlab="log(lambda)", ylab="Test Deviance", main="N=100, p=10, K=4")
lines((log(cvfit1$lambda)[c(1:to_run)]),dev_L, type="o", col="blue")
lines((log(cvfit2$lambda)[c(1:to_run)]),dev_E,type="o", col="green")

legend(-10, 2.5, legend=c("plasso", "lasso","elastic net"),
       col=c("red", "blue","green"), lty=1:2, cex=0.8)

abline(v= log(result_cv$lambda.min), lty = 3,col="red")
abline(v= log(cvfit1$lambda.min), lty = 3,col="blue")
abline(v= log(cvfit2$lambda.min), lty = 3,col="green")

############################################################


glmnet_pred1<-predict(fit3,X_Z_test,type="coefficient",s=cvfit1$lambda.min)

glmnet_pred2<-predict(fit4,X_Z_test,type="coefficient",s=cvfit2$lambda.min)


predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=lambda)
s_pliable<-data.frame(predict$beta)

glmnet_pred1
glmnet_pred2
s_pliable





predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=lambda)

glmnet_pred1<-predict(fit3,X_Z_test,type="response",s=cvfit1$lambda.min)

glmnet_pred2<-predict(fit4,X_Z_test,type="response",s=cvfit2$lambda.min)


s_pliable<-data.frame(predict$y_hat)


s_lasso<-data.frame(matrix(glmnet_pred1,N_test,max(y_test)))
s_elastic<-data.frame(matrix(glmnet_pred2,N_test,max(y_test)))
v_L=matrix(0,N_test)

v_P=matrix(0,N_test)
v_E=matrix(0,N_test)

g_P<-matrix(0,N_test)
g_L<-matrix(0,N_test)
g_E=matrix(0,N_test)




for (i in 1:N_test) {
  g_P[i]=max(s_pliable[i,])
  g_L[i]=max(s_lasso[i,])
  g_E[i]=max(s_elastic[i,])
  v_P[i]<- which(s_pliable[i,]==g_P[i])
  v_L[i]<-which(s_lasso[i,]==g_L[i])
  v_E[i]<- which(s_elastic[i,]==g_E[i])
}


val_lasso=matrix(0,N_test)
val_pliable=matrix(0,N_test)

val_elastic=matrix(0,N_test)



for (i in 1:N_test) {
  if(isTRUE( y_test[i]==v_P[i])){val_pliable[i]=1
  
  }else {
    val_pliable[i]=0
  }
  
}




for (i in 1:N_test) {
  if(isTRUE( y_test[i]==v_L[i])){val_lasso[i]=1
  
  }else {
    val_lasso[i]=0
  }
  
}

for (i in 1:N_test) {
  if(isTRUE( y_test[i]==v_E[i])){val_elastic[i]=1
  
  }else {
    val_elastic[i]=0
  }
  
}

c<-data.frame(y_test,v_P,v_L,v_E,val_pliable,val_lasso,val_elastic)



sum(val_lasso)
sum(val_pliable)
sum(val_elastic)
table(c$y_test,c$v_P)
table(c$y_test,c$v_L)
table(c$y_test,c$v_E)





tab_P<-matrix(0,to_run); tab_L<-matrix(0,to_run);tab_E<-matrix(0,to_run)
for (n in 1:to_run) {
  
  
  predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=Lambda[n])
  
  
  glmnet_pred1<-predict(fit3,X_Z_test,type="response",s=cvfit1$lambda[n])
  
  glmnet_pred2<-predict(fit4,X_Z_test,type="response",s=cvfit2$lambda[n])
  
  
  s_pliable<-data.frame(predict$y_hat)
  
  
  s_lasso<-data.frame(matrix(glmnet_pred1,N_test,max(y_test)))
  s_elastic<-data.frame(matrix(glmnet_pred2,N_test,max(y_test)))
  v_L=matrix(0,N_test)
  
  v_P=matrix(0,N_test)
  v_E=matrix(0,N_test)
  
  g_P<-matrix(0,N_test)
  g_L<-matrix(0,N_test)
  g_E=matrix(0,N_test)
  
  
  
  
  for (i in 1:N_test) {
    g_P[i]=max(s_pliable[i,])
    g_L[i]=max(s_lasso[i,])
    g_E[i]=max(s_elastic[i,])
    v_P[i]<- which(s_pliable[i,]==g_P[i])
    v_L[i]<-which(s_lasso[i,]==g_L[i])
    v_E[i]<- which(s_elastic[i,]==g_E[i])
  }
  
  
  val_lasso=matrix(0,N_test)
  val_pliable=matrix(0,N_test)
  
  val_elastic=matrix(0,N_test)
  
  
  
  for (i in 1:N_test) {
    if(isTRUE( y_test[i]==v_P[i])){val_pliable[i]=1
    
    }else {
      val_pliable[i]=0
    }
    
  }
  
  
  
  
  for (i in 1:N_test) {
    if(isTRUE( y_test[i]==v_L[i])){val_lasso[i]=1
    
    }else {
      val_lasso[i]=0
    }
    
  }
  
  for (i in 1:N_test) {
    if(isTRUE( y_test[i]==v_E[i])){val_elastic[i]=1
    
    }else {
      val_elastic[i]=0
    }
    
  }
  
  c<-data.frame(y_test,v_P,v_L,v_E,val_pliable,val_lasso,val_elastic)
  
  
  
  tab_L[n]<- sum(val_lasso)
  tab_P[n]<-sum(val_pliable)
  tab_E[n]<-  sum(val_elastic)
  
}


test_tab<- data.frame(plasso=tab_P,lasso=tab_L,E_net=tab_E)
boxplot(test_tab,ylab="predicted values")

plot(log(as.matrix(result_cv$lambda)[c(1:to_run)]),tab_P,type="o", col="red",ylim=c(min(c(tab_P,tab_L,dev_E)),max(c(tab_P,tab_L,tab_E))),xlab="log(lambda)", ylab="Number of True prediction", main="N=100, p=10, K=4",xlim =c(min(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))),max(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))) ))
lines(log(cvfit1$lambda)[c(1:to_run)],tab_L, type="o", col="blue")
lines(log(cvfit2$lambda)[c(1:to_run)],tab_E,type="o", col="green")

legend(-10, 200, legend=c("plasso", "lasso","elastic net"),
       col=c("red", "blue","green"), lty=1:2, cex=0.8)


abline(v= log(result_cv$lambda.min), lty = 3,col="red")
abline(v= log(cvfit1$lambda.min), lty = 3,col="blue")
abline(v= log(cvfit2$lambda.min), lty = 3,col="green")

################################################################



#TRIAL 1
library(glmnet)



N = 100 ; p =10;nz=4; K=nz
#X <- matrix(rnorm(n = N * p, mean = 0, sd = 1), nrow = N, ncol = p)
#sim1_X<-X
#Z <- matrix(rbinom(n = N * K, size = 1, prob = 0.5), nrow = N, ncol = K)
#sim1_Z<-Z
X <- matrix(rnorm( N * p), nrow = N, ncol = p)

mx=colMeans(X)

sx=sqrt(apply(X,2,var))
X=scale(X,mx,sx)

X=matrix(as.numeric(X),N,p)

#Z <- matrix(rbinom(n = N * K, size = 1, prob = 0.5), nrow = N, ncol = K)
Z =matrix(rnorm(N*nz),N,nz)
mz=colMeans(Z)
sz=sqrt(apply(Z,2,var))

Z=scale(Z,mz,sz)
e=matrix(1,N)
beta_1 <- rep(x = 0, times = p); beta_2<-rep(x = 0, times = p);beta_3<-rep(x = 0, times = p)
beta_1[1:5] <- c(2, 2, 2, 2,2); beta_2[1:5]<-c(-2,  -2,-2, -2,-2); beta_3[1:5]<-c(4, -2, 1, 3,2)
#beta_1[1:5]<-c(4,-2,2,2,-5)
coeffs1 <- cbind(beta_1[1]+5*Z[,1], beta_1[2], beta_1[3] +  3*Z[, 2],  beta_1[4] *(e -  2*Z[, 3]),beta_1[5]*(e-2*Z[,4]))+.5*rnorm(N)
coeffs2 <- cbind(beta_2[1]+5*Z[,1], beta_2[2], beta_2[3] +  3*Z[, 2],  beta_2[4] *(e -  2*Z[, 3]),beta_2[5]*(e-2*Z[,4]))+.5*rnorm(N)
coeffs3 <- cbind(beta_3[1]+5*Z[,1], beta_3[2], beta_3[3] +  3*Z[, 2],  beta_3[4] *(e -  2*Z[, 3]),beta_3[5]*(e-2*Z[,4]))+.5*rnorm(N)





#c1=matrix(0,N,p);c2=matrix(0,N,p);c3=matrix(0,N,p)
#c1[,c(1:5)]<-coeffs1;c2[,c(1:5)]<-coeffs2;c3[,c(1:5)]<-coeffs3

# 
# coeffs1 <- cbind(beta_1[1],beta_1[2], beta_1[3] -  3*Z[, 1]+3*Z[,2],  beta_1[4]  -  2*Z[, 3]+2*Z[,4],beta_1[5])
#coeffs2 <- cbind(beta_2[1],beta_2[2], beta_2[3] -  3*Z[, 1]+3*Z[,2],  beta_2[4]  -  2*Z[, 3]+2*Z[,4],beta_2[5])
# 
# coeffs3 <- cbind(beta_3[1],beta_3[2], beta_3[3] -  3*Z[, 1]+3*Z[,2],  beta_3[4]  -  2*Z[, 3]+2*Z[,4],beta_3[5])



vProb = cbind(exp(diag(X[, 1:5]%*%t(coeffs1))), exp(diag(X[, 1:5]%*%t(coeffs2))), exp(diag(X[, 1:5]%*%t(coeffs3))))
mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
y = apply(mChoices, 1, function(x) which(x==1))
num_eff=5
#y_1<-4*X[,1] -2*X[,2]+2*X[,3] +2*X[,4]-5*X[,5]-2*X[,1]*Z[,1]+-2*X[,1]*Z[,2]+X[,2]*(4*Z[,3]-Z[,4])+ 4*rnorm(N)

#y1=1*(y_1>=0 & y_1<=3)
#y2=2*(y_1<0)
#y3=3*(y_1>3)
#e=matrix(y1+y2+y3)
#Z=matrix(as.numeric(Z),N,nz)
#y=e
table(y)
#dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), X)
#y=gendata_MLR(N, p, NC = 3, X = X, betas = coeffs1)


#mu <- 4*X[,1] -2*X[,2]+X[,3] +5*X[,1]*Z[,1]+3*X[,3]*Z[,2]+X[,4]*(3-2*Z[,3])+ 3*rnorm(N )  #diag(X[, 1:5] %*% t(coeffs))
#y <- mu + 4 * rnorm(N, mean = 0, sd = 1)


#y =mu + 0.5 * rnorm(N, mean = 0, sd = 1)

#y1=1*(y>= -1 & y<=3)


#y2=2*(y< -1)

#y3=3*(y>3)


#e=matrix(y1+y2+y3)


#Z=matrix(as.numeric(Z),N,nz)
#y=e
#table(y)


nlambda=50

#system.time(result <- plasso_fit1(y=y, X, Z, nlambda, alpha = .5,new_t=1.,my_mbeta=-.02,intercept = 0.001,step=-.05,number=0,maxgrid=50,tol= -1.4e-1,run=2))
tt=rep(t,nlambda)
for_v=10;sv=0;fq=50;st=50;mv=20;ms=50;tol=1e-3
system.time(result <- plasso_fit1(y=y, X, Z, nlambda, alpha = .5,new_t=1,my_mbeta=.09,intercept = 0.01,step=.05,number=10,maxgrid=50,tol= tol,run=2,lambda_min = .001,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,cv_run = 0,max_iter=10000))



#p_lasso=pliableMulti(y=y,X,Z,nlambda = nlambda,alpha=.5,family = "multinomial")

path<-result$path

path


big_t<-result$big_T
Mbeta=result$Mbeta
Lambda<-as.matrix(result$Lambdas)
glmnet.control(fdev = 0)
# 
#fit1<-glmnet(x=X,y=y,alpha = 1,family = "multinomial" ,nlambda =50)
# 
# fit2<-glmnet(x=X,y=y,alpha = 0.5,family = "multinomial" ,nlambda = 50)
#data_gbm<-data.frame(Y=y,X)
#fit3$estimator

dat <- data.frame( X, Z)
names(dat) <- c(paste0("X", 1:ncol(X)),paste0("Z", 1:ncol(Z)) )
#f <- as.formula(y ~ .*.)
#f <- as.formula(y~ (X1 +X1:Z1)+X2+(X3+X3:Z2)+(X4-X4:Z3)+(X5-X5:Z4)+X6+X7+X8+X9+X10  +0)
#cbind(beta_1[1]+5*Z[,1], beta_1[2], beta_1[3] +  3*Z[, 2],  beta_1[4] *(e -  2*Z[, 3]),beta_1[5]*(e-2*Z[,4]))+.5*rnorm(N)

y_Z <- y
X_Z <- as.matrix(dat)#model.matrix(f, data=dat)[, -1]


fit3<-glmnet(x=X_Z,y=y_Z,alpha = 1,family = "multinomial" ,nlambda =50)

fit4<-glmnet(x=X_Z,y=y_Z,alpha = 0.5,family = "multinomial" ,nlambda = 50)


fitdiv1<-deviance(fit3)/length(y)
fitdiv2<-deviance(fit4)/length(y)


plot(log(fit3$lambda),fitdiv1, type="o", col="red",xlab = "log(lambda)", ylab = "Train Deviance",ylim =c(min((c(path$DEV,fitdiv1,fitdiv2))), max((c(path$DEV,fitdiv1,fitdiv2)))),xlim =c(min(c(log(as.matrix(Lambda)),log(fit3$lambda),log(fit4$lambda))),max(c(log(as.matrix(Lambda)),log(fit3$lambda),log(fit4$lambda)))))
lines(log(Lambda)[c(1:50)],path$DEV[c(1:50)], col="blue",type = "o")
#lines(log(fit2$lambda)[c(1:50)],fit2$dev.ratio[c(1:50)], type="o", col="green")
#lines(log(fit3$lambda)[c(1:50)],fit3$dev.ratio[c(1:50)], type="o", col="yellow")
lines(log(fit4$lambda)[c(1:50)],fitdiv2[c(1:50)], type="o", col="green")


legend(-10, 2, legend=c("lasso", "plasso","elastic net"),
       col=c("red", "blue","green"), lty=1:2, cex=0.8)


data<-data.frame(M_pliable=path$DEV[c(1:50)],lasso=fitdiv1,elastic_net=fitdiv2[c(1:50)])
boxplot(data, main="N=100, p=10, K=4", ylab="Train Deviance")

plot(log(Lambda),path$DEV,type="o", col="red",xlab = "log(lambda)", ylab = " Deviance")







# 
# 
# Coeff<-lapply(seq_len(max(y)),
#               function(j)(matrix(0,ncol(X),nlambda)))
# 
# 
# for (i in 1:nlambda) {
#   f<-matrix(unlist(result$beta[i]),ncol(X),max(y))
#   for (j in 1:max(y)) {
#     Coeff[[j]][,i]<-f[,j]
#   }
#   
# }
# 



for (r in 1:max(y)) {
  # q<-matrix(unlist(Coeff[[i]]),ncol(X),nlambda)
  
  gg=nlambda
  my_beta<-matrix(0,p,gg)
  
  
  #result$beta[[r]][1,]
  for (i in 1:p) {
    
    
    my_beta[i,]<-result$beta[[r]][i,]
    
    
  }
  
  
  b=apply(abs(my_beta), 2, sum)
  b=c(1:gg)
  n=dim(my_beta)[2]
  matplot(b,cbind( t(my_beta),fit$path$OBJ_main[c(1:gg)]),type="n",col="red", ylim = range(my_beta),  xlab="nlambda", ylab= ( paste(  "Coefficients", r)) )
  
  #plot(b,result$beta[[r]][1,],type="l",col="red", ylim = range(result$beta[[r]]), ylab=paste(  "Coefficients", r)   ,xlab = "nlambda")
  axis(side = 3, at =  (as.matrix(b)), labels = paste(as.matrix(result$non_zero[,r])),
       tick = FALSE, line = 0)
  
  for (i in 1:num_eff) {
    lines(b,(my_beta[i,]),col=i+1, lty=1  )
    
    text( (b[n]+1),my_beta[i,n], labels = i)
  }
  for (i in (num_eff+1):p) {
    lines(b,(my_beta[i,]),col=i+1, lty=2  )
    
    text( (b[n]+1),my_beta[i,n], labels = i)
  }
  
  
  
  
  act=which(rowSums(abs(my_beta))>0)
  theta1<-array(0,c(ncol(X),ncol(Z),(nlambda) ))
  for (i in 1:(nlambda)) {
    theta1[,,i]<-matrix(unlist(result$theta[[r]][,,i]),p,K)
  }
  
  
  ntheta=  apply( abs(theta1)>0,c(1,3),sum)
  index = c(1:gg)#apply(abs(my_beta), 2, sum)+apply(abs(theta1),3,sum)
  sbeta=(my_beta)
  for(j in act){
    for(i in 1:length(index)){
      
      if(ntheta[j,i]>0) text(index[i],sbeta[j,i],label="x",cex=.7)
    }}
  
  
  
  
  
}

#plot(fit1,label=T)

#plot(fit2,label=T)

plot(fit3,label=T)

plot(fit4,label=T)


####cv for trial 1
####### Z trained########
nfolds=5
set.seed(3321)
foldid = sample(rep(seq(nfolds), length = N))
foldid = sample(rep(1:nfolds, ceiling(N/nfolds)), N, replace=FALSE)
# predict z  from x; here we use glmnet, but any other supervised learning procedure
#  could be used
zhat=matrix(NA,N,ncol(Z))
for(ii in 1:nfolds){
  zfit=cv.glmnet(X[foldid!=ii,],Z[foldid!=ii,],family="mgaussian")
  zhat[foldid==ii,]=predict(zfit,X[foldid==ii,],s=zfit$lambda.min)
}

#NOTE that the same foldid vector must be passed to cv.pliable
#this is used when you want to predict Z from X
#cvfit=cv.pliable(fit,x,zhat,y,foldid=foldid)
#plot(cvfit)
####### ###################
system.time(result_cv <- cv_mpliable(fit=result,nfolds=5,X,Z=zhat,y,alpha=.5,new_t=.5,nlambda=50,maxgrid=50,tol=tol,lambda_min=0.0001,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,foldid=foldid),max_iter=1000)


result_cv


#result_cv. This used when you want to generate random values for Z
###################################################

system.time(result_cv <- cv_mpliable(fit=result,nfolds=5,X,Z,y,alpha=.5,new_t=.5,nlambda=50,maxgrid=50,tol=tol,lambda_min=0.0001,for_v=for_v,sv=sv,fq=fq,st=st,mv=mv,ms=ms,max_iter=1000))
#system.time(result_cv <- cv_mpliable(fit=result,nfolds=5,X,Z,y,alpha=.5,nlambda=50,maxgrid=50,for_v=for_v,sv=sv,fq=fq,st=st))


result_cv

#result_cv




cvfit1<-cv.glmnet(x=X_Z,y=y_Z,alpha = 1,family = "multinomial", nfolds = 5,nlambda=nlambda)


cvfit2<-cv.glmnet(x=X_Z,y=y_Z,alpha = .5,family = "multinomial", nfolds = 5,nlambda=nlambda)
#deviance from (test samples? all samples?) #at lambda value


plot(cvfit1)
plot(cvfit2)





plot(log(as.matrix(result_cv$lambda)),(as.matrix(result_cv$cvm)), type = 'o', xlab = "log(lambda)",ylab = "Deviance",pch = 20,col="red",ylim=c(min(c(result_cv$cvm,(result_cv$cvup)[c(1:length(result_cv$cvm))],result_cv$cvlo[c(1:length(result_cv$cvm))] )),max(c(result_cv$cvm,(result_cv$cvup)[c(1:length(result_cv$cvm))],result_cv$cvlo[c(1:length(result_cv$cvm))])))  )

abline(v= log(result_cv$lambda.min), lty = 3)
#abline(v= log(plasso_cv$lambda.min), lty = 3)
abline(v= log(result_cv$lambda.1se), lty = 3)
axis(side = 3, at =  log(as.matrix(result_cv$lambda)), labels = paste(as.matrix(result_cv$nz)),
     tick = FALSE, line = 0)


error.bars( log(result_cv$lambda), result_cv$cvup, result_cv$cvlo,
            width = 0.01, col = "darkgrey")





lambda<-as.numeric(result_cv$lambda.min)


#z1<-1*(Z[,1]==1);z2<-2*(Z[,2]==1);z3<-3*(Z[,3]==1);z4<-4*(Z[,4]==1)
#my_Z<-z1+z2+z3+z4
#to predict Z form X
#note: if you don't use this , then Z_test must be genrated same way as X_test
zfit<-cv.glmnet(X,zhat,family="mgaussian")

######################3#######################3##############3#
######################PREDICT#####################3##########

N_test = 500 ; p =p;nz=nz; K=nz
#X_test <- matrix(rnorm(n = N_test * p, mean = 0, sd = 1), nrow = N_test, ncol = p)
X_test <- matrix(rnorm( N_test * p), nrow = N_test, ncol = p)

#mx=colMeans(X_test)

#sx=sqrt(apply(X_test,2,var))
X_test=scale(X_test,mx,sx)
#X_test=matrix(as.numeric(X_test),N_test,p)

#to predict Z_test
Z_test =predict(zfit,X_test,s=zfit$lambda.min)[,,1]


# to generate random Z
# 
Z_test =matrix(rnorm(N_test*nz),N_test,nz)
mz=colMeans(Z_test)
sz=sqrt(apply(Z_test,2,var))

Z_test=scale(Z_test,mz,sz)
#Z_test <- matrix(rbinom(n = N_test * K, size = 1, prob = 0.5), nrow = N_test, ncol = K)
#Z_test=zhat
e=matrix(1,N_test)
beta_1 <- rep(x = 0, times = p); beta_2<-rep(x = 0, times = p);beta_3<-rep(x = 0, times = p)
beta_1[1:5] <- c(2, 2, 2, 2,2); beta_2[1:5]<-c(-2,  -2,-2, -2,-2); beta_3[1:5]<-c(4, -2, 1, 3,2)
# beta_1[1:5]<-c(4,-2,2,2,-5)
coeffs1 <- cbind(beta_1[1]+5*Z_test[,1], beta_1[2], beta_1[3] +  3*Z_test[, 2],  beta_1[4] *(e -  2*Z_test[, 3]),beta_1[5]*(e-2*Z_test[,4]))+.5*rnorm(N)
coeffs2 <- cbind(beta_2[1]+5*Z_test[,1], beta_2[2], beta_2[3] +  3*Z_test[, 2],  beta_2[4] *(e -  2*Z_test[, 3]),beta_2[5]*(e-2*Z_test[,4]))+.5*rnorm(N)
coeffs3 <- cbind(beta_3[1]+5*Z_test[,1], beta_3[2], beta_3[3] +  3*Z_test[, 2],  beta_3[4] *(e -  2*Z_test[, 3]),beta_3[5]*(e-2*Z_test[,4]))+.5*rnorm(N)





# 
# coeffs1 <- cbind(beta_1[1], beta_1[2], beta_1[3] -  3*Z_test[, 1]+3*Z_test[, 2],  beta_1[4]  -  2*Z_test[, 3]+2*Z_test[,4],beta_1[5])
# coeffs2 <-cbind(beta_2[1], beta_2[2], beta_2[3] -  3*Z_test[, 1]+3*Z_test[, 2],  beta_2[4]  -  2*Z_test[, 3]+2*Z_test[,4],beta_2[5])
# coeffs3 <- cbind(beta_3[1], beta_3[2], beta_3[3] -  3*Z_test[, 1]+3*Z_test[, 2],  beta_3[4]  -  2*Z_test[, 3]+2*Z_test[,4],beta_3[5])

vProb = cbind(exp(diag(X_test[, 1:5]%*%t(coeffs1))), exp(diag(X_test[, 1:5]%*%t(coeffs2))), exp(diag(X_test[, 1:5]%*%t(coeffs3))))
mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
y_test = apply(mChoices, 1, function(x) which(x==1))






table(y_test)





#Z_test=matrix(as.numeric(Z_test),N_test,nz)




dat_test <- data.frame( X_test, Z_test)
names(dat_test) <- c(paste0("X", 1:ncol(X_test)),paste0("Z", 1:ncol(Z_test)))
#f <- as.formula(y~X1 +X1:Z1+X2+X3+X3:Z2+X4-X4:Z3+X5-X5:Z4  +0)
y_Z_test <- y_test
#X_Z_test<- model.matrix(f, dat_test)[, -1]

X_Z_test<-as.matrix(dat_test)



to_run=50

dev_P<-matrix(0,to_run); dev_L<-matrix(0,to_run);dev_E<-matrix(0,to_run)
for (n in 1:to_run) {
  
  
  predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=Lambda[n])
  dev_P[n]<-predict$deviance
  
  glmnet_pred1<-predict(fit3,newx=X_Z_test,type="response",s=cvfit1$lambda[n])
  pr_l<- data.frame(glmnet_pred1)
  glmnet_pred2<-predict(fit4,newx=X_Z_test,type="response",s=cvfit2$lambda[n])
  pr_e<- data.frame(glmnet_pred2)
  
  Dev_l<-matrix(0,nrow = length(y_test))
  Dev_e<-matrix(0,nrow = length(y_test))
  for (l in 1:length(Dev_l)) {
    deviance_y1<-matrix(0,1,max(y_test))
    deviance_y2<-matrix(0,1,max(y_test))
    for (d in 1:max(y_test)) {
      y_d<-1*(y_test[l]==d)
      # n_i_d<-matrix(unlist(n_i[d]),N)
      prob_d_l <-matrix(unlist(pr_l[,d]),N_test)
      prob_d_e <-matrix(unlist(pr_e[,d]),N_test)
      # deviance_y1[d]<-y_d*n_i_d[l]
      # deviance_y2[d]<-exp(n_i_d[l])
      deviance_y1[d]<-  Log(y=y_d,pr=prob_d_l[l])
      deviance_y2[d]<-  Log(y=y_d,pr=prob_d_e[l]) 
      #+(1-y_d)*Log(y=(1-y_d),pr=(1-prob_d[l]))
    }
    Dev_l[l]<-  sum(deviance_y1)
    Dev_e[l]<-  sum(deviance_y2)
  }
  
  # Dev_l<-matrix(0,nrow = length(y_test))
  # Dev_e<-matrix(0,nrow = length(y_test))
  # for (l in 1:length(Dev_l)) {
  #   deviance_y1<-matrix(0,1,max(y_test))
  #   deviance_y2<-matrix(0,1,max(y_test))
  #   for (d in 1:max(y_test)) {
  #     y_d<-1*(y_test[l]==d)
  #     # n_i_d<-matrix(unlist(n_i[d]),N)
  #     prob_d_l <-matrix(unlist(pr_l[,d]),N_test)
  #     prob_d_e <-matrix(unlist(pr_e[,d]),N_test)
  #     # deviance_y1[d]<-y_d*n_i_d[l]
  #     # deviance_y2[d]<-exp(n_i_d[l])
  #     deviance_y1[d]<-sum(errfun.binomial(y=y_d,yhat=prob_d_l[l],w=rep(1,length(y_d)))) #+(1-y_d)*Log(y=(1-y_d),pr=(1-prob_d[l]))
  #     #print(deviance_y1[d])
  #     deviance_y2[d]=sum(errfun.binomial(y=y_d,yhat=prob_d_e[l],w=rep(1,length(y_d))))
  #   }
  #   Dev_l[l]<-  sum(deviance_y1)
  #   Dev_e[l]<-  sum(deviance_y2)
  #   
  #   
  #   #(y,yhat,w=rep(1,length(y)))
  # }
  dev_L[n]<-((2)*sum(Dev_l))/length(y_test)
  dev_E[n]<-((2)*sum(Dev_e))/length(y_test)
  
  
  
  #dev_L[n]<- ((2)*sum(Dev_l))#
  #dev_E[n]<- ((2)*sum(Dev_e))#
}

test_deviance<- data.frame(plasso=dev_P,lasso=dev_L,E_net=dev_E)
boxplot(test_deviance,ylab="Test deviance")
##########################################################
plot(log(as.matrix(Lambda)[c(1:to_run)]),dev_P,type="o", col="red",ylim=c(min(c(dev_P,dev_L,dev_E)),max(c(dev_P,dev_L,dev_E))),xlim =c(min(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))),max(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))) ) ,xlab="log(lambda)", ylab="Test Deviance", main="N=100, p=10, K=4")
lines((log(cvfit1$lambda)[c(1:to_run)]),dev_L, type="o", col="blue")
lines((log(cvfit2$lambda)[c(1:to_run)]),dev_E,type="o", col="green")

legend(-10, 2.5, legend=c("plasso", "lasso","elastic net"),
       col=c("red", "blue","green"), lty=1:2, cex=0.8)

abline(v= log(result_cv$lambda.min), lty = 3,col="red")
abline(v= log(cvfit1$lambda.min), lty = 3,col="blue")
abline(v= log(cvfit2$lambda.min), lty = 3,col="green")

############################################################


glmnet_pred1<-predict(fit3,X_Z_test,type="coefficient",s=cvfit1$lambda.min)

glmnet_pred2<-predict(fit4,X_Z_test,type="coefficient",s=cvfit2$lambda.min)


predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=lambda)
s_pliable<-data.frame(predict$beta)

glmnet_pred1
glmnet_pred2
s_pliable





predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=lambda)

glmnet_pred1<-predict(fit3,X_Z_test,type="response",s=cvfit1$lambda.min)

glmnet_pred2<-predict(fit4,X_Z_test,type="response",s=cvfit2$lambda.min)


s_pliable<-data.frame(predict$y_hat)


s_lasso<-data.frame(matrix(glmnet_pred1,N_test,max(y_test)))
s_elastic<-data.frame(matrix(glmnet_pred2,N_test,max(y_test)))
v_L=matrix(0,N_test)

v_P=matrix(0,N_test)
v_E=matrix(0,N_test)

g_P<-matrix(0,N_test)
g_L<-matrix(0,N_test)
g_E=matrix(0,N_test)




for (i in 1:N_test) {
  g_P[i]=max(s_pliable[i,])
  g_L[i]=max(s_lasso[i,])
  g_E[i]=max(s_elastic[i,])
  v_P[i]<- which(s_pliable[i,]==g_P[i])
  v_L[i]<-which(s_lasso[i,]==g_L[i])
  v_E[i]<- which(s_elastic[i,]==g_E[i])
}


val_lasso=matrix(0,N_test)
val_pliable=matrix(0,N_test)

val_elastic=matrix(0,N_test)



for (i in 1:N_test) {
  if(isTRUE( y_test[i]==v_P[i])){val_pliable[i]=1
  
  }else {
    val_pliable[i]=0
  }
  
}




for (i in 1:N_test) {
  if(isTRUE( y_test[i]==v_L[i])){val_lasso[i]=1
  
  }else {
    val_lasso[i]=0
  }
  
}

for (i in 1:N_test) {
  if(isTRUE( y_test[i]==v_E[i])){val_elastic[i]=1
  
  }else {
    val_elastic[i]=0
  }
  
}

c<-data.frame(y_test,v_P,v_L,v_E,val_pliable,val_lasso,val_elastic)



sum(val_lasso)
sum(val_pliable)
sum(val_elastic)
table(c$y_test,c$v_P)
table(c$y_test,c$v_L)
table(c$y_test,c$v_E)





tab_P<-matrix(0,to_run); tab_L<-matrix(0,to_run);tab_E<-matrix(0,to_run)
for (n in 1:to_run) {
  
  
  predict<-predict_lasso(result ,X=X_test,Z=Z_test,y=y_test,lambda=Lambda[n])
  
  
  glmnet_pred1<-predict(fit3,X_Z_test,type="response",s=cvfit1$lambda[n])
  
  glmnet_pred2<-predict(fit4,X_Z_test,type="response",s=cvfit2$lambda[n])
  
  
  s_pliable<-data.frame(predict$y_hat)
  
  
  s_lasso<-data.frame(matrix(glmnet_pred1,N_test,max(y_test)))
  s_elastic<-data.frame(matrix(glmnet_pred2,N_test,max(y_test)))
  v_L=matrix(0,N_test)
  
  v_P=matrix(0,N_test)
  v_E=matrix(0,N_test)
  
  g_P<-matrix(0,N_test)
  g_L<-matrix(0,N_test)
  g_E=matrix(0,N_test)
  
  
  
  
  for (i in 1:N_test) {
    g_P[i]=max(s_pliable[i,])
    g_L[i]=max(s_lasso[i,])
    g_E[i]=max(s_elastic[i,])
    v_P[i]<- which(s_pliable[i,]==g_P[i])
    v_L[i]<-which(s_lasso[i,]==g_L[i])
    v_E[i]<- which(s_elastic[i,]==g_E[i])
  }
  
  
  val_lasso=matrix(0,N_test)
  val_pliable=matrix(0,N_test)
  
  val_elastic=matrix(0,N_test)
  
  
  
  for (i in 1:N_test) {
    if(isTRUE( y_test[i]==v_P[i])){val_pliable[i]=1
    
    }else {
      val_pliable[i]=0
    }
    
  }
  
  
  
  
  for (i in 1:N_test) {
    if(isTRUE( y_test[i]==v_L[i])){val_lasso[i]=1
    
    }else {
      val_lasso[i]=0
    }
    
  }
  
  for (i in 1:N_test) {
    if(isTRUE( y_test[i]==v_E[i])){val_elastic[i]=1
    
    }else {
      val_elastic[i]=0
    }
    
  }
  
  c<-data.frame(y_test,v_P,v_L,v_E,val_pliable,val_lasso,val_elastic)
  
  
  
  tab_L[n]<- sum(val_lasso)
  tab_P[n]<-sum(val_pliable)
  tab_E[n]<-  sum(val_elastic)
  
}


test_tab<- data.frame(plasso=tab_P,lasso=tab_L,E_net=tab_E)
boxplot(test_tab,ylab="predicted values")

plot(log(as.matrix(result_cv$lambda)[c(1:to_run)]),tab_P,type="o", col="red",ylim=c(min(c(tab_P,tab_L,dev_E)),max(c(tab_P,tab_L,tab_E))),xlab="log(lambda)", ylab="Number of True prediction", main="N=100, p=10, K=4",xlim =c(min(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))),max(c(log(as.matrix(Lambda)[c(1:to_run)]),(log(cvfit1$lambda)[c(1:to_run)]),(log(cvfit2$lambda)[c(1:to_run)]))) ))
lines(log(cvfit1$lambda)[c(1:to_run)],tab_L, type="o", col="blue")
lines(log(cvfit2$lambda)[c(1:to_run)],tab_E,type="o", col="green")

legend(-10, 200, legend=c("plasso", "lasso","elastic net"),
       col=c("red", "blue","green"), lty=1:2, cex=0.8)


abline(v= log(result_cv$lambda.min), lty = 3,col="red")
abline(v= log(cvfit1$lambda.min), lty = 3,col="blue")
abline(v= log(cvfit2$lambda.min), lty = 3,col="green")

################################################################

