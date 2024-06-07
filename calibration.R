id_number = 1
setwd("/Users/jiwoo/Downloads/Code/Yonsei/ICT")
original_data = read.csv("original_data/data_id1.csv")
file_list = list.files(paste0("aug_synth_data/apt_", id_number), full.names = TRUE)
file_list
aug_synthetic_data = read.csv(file_list[1])
head(original_data)
head(aug_synthetic_data)
summary(original_data)
summary(aug_synthetic_data)
nrow(original_data)
nrow(aug_synthetic_data)

#original_data = original_data[original_data['Apartment'] == '공덕',3:6]
#selected_rows <- original_data[original_data$April >= 0 & original_data$May >= 0 & original_data$June >= 0 & original_data$July >= 0, ]
#original_data <- original_data[order(row.names(selected_rows)), ]
#rownames(original_data) <- NULL


{ # synthetic data
  m = nrow(aug_synthetic_data)
  A = as.matrix(aug_synthetic_data)
  weights = rep(1,m)
}

{ # benchmark information
  desired_means = colMeans(original_data) # c(2,2) # benchmark of mean: (1,1)
}

{ # to find lambda, f(lambda)=0
  lambda = matrix(rep(0, length(desired_means)), ncol=1) # initial lambda 
  
  diff = function(lambda, weights, A){
    weights_prev = weights
    weights_post = weights_prev * exp(-A %*% lambda)
    weights_post = weights_post/sum(weights_post)*m
    
    return(t(A) %*% weights_post / m - desired_means)
  } # the function 'f'
  
  { # Newton-Raphson algo, lambda* = lambda0 - f'(lambda0)f(lambda0)
    eps = matrix(c(1,1), ncol=1)
    while(sum(abs(eps))>1e-6){
      # eps := f'(lambda0)f(lambda0)
      eps = solve(numDeriv::jacobian(function(lambda)diff(lambda, weights=weights, A=A), lambda)) %*% diff(lambda, weights, A)
      print("eps")
      print(eps)
      lambda = lambda - eps; print(sum(abs(eps)))
      print(lambda)
    }
  }
}
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

synthetic_data = aug_synthetic_data[sample(1:m, nrow(original_data), prob=weights_calib, replace = T),]
synthetic_data
nrow(synthetic_data)
summary(original_data)
summary(synthetic_data)
write.csv(synthetic_data, "/Users/jiwoo/Downloads/Code/Yonsei/ICT/synthetic_data/our_method/apt_1/data2.csv", row.names=FALSE)


om_data = read.csv("smartgrid_timeseries_1.csv")
om_data = om_data[om_data['Apartment'] == '공덕',4:7]
om_data = om_data[om_data$April >= 0 & om_data$May >= 0 & om_data$June >= 0 & om_data$July >= 0, ]
nrow(om_data)
summary(om_data)
original_std_deviation <- apply(original_data, 2, sd)
synthetic_std_deviation <- apply(synthetic_data, 2, sd)
om_std_deviation <- apply(om_data, 2, sd)
print(original_std_deviation)
print(synthetic_std_deviation)
print(om_std_deviation)

write.csv(synthetic_data, file = "final_synth_data/final_synthetic_data_id7.csv", row.names = FALSE)
