id_number = 1

original_data = read.csv("C:/Users/wjswp/Desktop/발표자료/simulation/ds_energy/diffusion-ts/original_data.csv")
aug_synthetic_data = read.csv('C:/Users/wjswp/Desktop/발표자료/simulation/ds_energy/diffusion-ts/syn_ds.csv')

original_data=t(original_data)
#original_data = original_data[original_data['Apartment'] == '공덕',3:6]
#selected_rows <- original_data[original_data$April >= 0 & original_data$May >= 0 & original_data$June >= 0 & original_data$July >= 0, ]
#original_data <- original_data[order(row.names(selected_rows)), ]
rownames(original_data) <- NULL



{ # synthetic data
  l=length(t(aug_synthetic_data))/length(t(original_data))
  m=nrow(aug_synthetic_data)
  A=as.matrix(aug_synthetic_data)
  n=nrow(original_data)
  weights=rep(1,m)/l
}

{ # benchmark information
  desired_means=colMeans(original_data)
  #평균을 벤치마크로 하여 맞춰주고자 함
}

{ # function_diff
  lambda=matrix(rep(0,length(desired_means)),ncol=1)
  #초기 람다값=0
  
  diff = function(lambda, weights, A,desired_means){
    weights_prev = weights
    weights_post = weights_prev * exp(-A %*% lambda)
    weights_post = weights_post/sum(weights_post)*m
    
    return(t(A) %*% weights_post / m - desired_means)
  } # the function 'f'
  
  diff1=function(lambda,weights,A,desired_means,i){
    lambda_i<-lambda[i]
    A_i<-A[,i]
    desired_means_i<-desired_means[i]
    weights_prev=weights
    weights_post=weights_prev*exp(-A_i*lambda_i)
    weights_post=weights_post/sum(weights_post)*m
    
    return(as.vector(t(A_i)%*%weights_post)/m-desired_means_i)
  }
}





{ # to find lambda, f(lambda)=0
  # Newton-Raphson algo, albmda*=lambda0-f'(lambda0)f(lambda0)
  eps_tot=matrix(c(1,1),ncol=1)
  j=0
  while(sum(abs(eps_tot))>(1e-4)){
    eps_tot = solve(numDeriv::jacobian(function(lambda)diff(lambda, weights=weights, A=A,desired_means), lambda)) %*% diff(lambda, weights, A,desired_means)
    for (i in 1:12){
      eps=matrix(c(1,1),ncol=1)
      while(sum(abs(eps))>1e-10){
        eps=diff1(lambda,weights,A,desired_means,i)/numDeriv::jacobian(function(lambda){diff1(lambda,weights,A,desired_means,i)},lambda)[i]
        print('eps')
        print(eps)
        lambda[i]=lambda[i]-1*eps
        weights_calib=weights*exp(-A[,i]*lambda[i])
        weights = weights_calib/sum(weights_calib)*m
        #plot(weights,main=paste('iteration',i))
        #print(sum(abs(abs(matrix(weights,nrow=1) %*% as.matrix(aug_synthetic_data)-colMeans(original_data)))))
        #print(lambda)
        print('error')
        print(sum(abs(diff1(lambda,weights,A,desired_means,i))))
        print('eps_tot')
        print(sum(abs(eps_tot)))
      }
    print(i)
    }  
  }
}


### nleqslv 패키지 써서 다시해보기
library(nleqslv)
objective_function<-function(lambda){
  return(diff(lambda,weights,A,desired_means))
}
solution<-nleqslv(lambda,objective_function,jac=NULL,control=list(allowSingular=TRUE,maxit=5000))
solution
lambda<-solution$x






{ # calculate the calibration weights finally
  weights_calib = weights * exp(-A %*% lambda)
  weights_calib = weights_calib/sum(weights_calib)*m
}


{ # check for calibration code to work well
  t(A) %*% weights_calib / m
}

rst = sapply(1:100, function(iter){
  index= sample(1:m, nrow(aug_synthetic_data), prob=weights_calib, replace = T)
  colMeans(A[index,])
})
desired_means
rowMeans(rst)
rst
synthetic_data = aug_synthetic_data[sample(1:m, nrow(original_data), prob=weights_calib, replace = T),]


aug_synthetic_data
synthetic_data
nrow(synthetic_data)
summary(original_data)
summary(synthetic_data)
summary(t(rst))

original_std_deviation <- apply(original_data, 2, sd)
synthetic_std_deviation <- apply(synthetic_data, 2, sd)
om_std_deviation <- apply(original_data, 2, sd)
print(original_std_deviation)
print(synthetic_std_deviation)

t(rst)
write.csv(t(rst), file = "C:/Users/wjswp/Desktop/발표자료/simulation/ds_energy/diffusion-ts/synthetic_data_ds__.csv", row.names = FALSE)
