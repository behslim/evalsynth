# library(torch)
# library(R6)
# library(logging)
# library(dplyr)
# library(MASS)
# library(corpcor)
# library(FNN)
# library(synthpop)
# library(rpart)
# library(cluster)
# library(tidyverse)
# library(doParallel)
# library(ks)

##################################################
# distance_measures.R
##################################################
options(warning = -1)
device = 'cpu' # matrices are too big for gpu
torch::torch_manual_seed(1)

# This distance measures code is only applicable to numerical variables due to computation of WD.

column_prob_cont = function(orig, synt, nbins = 100){

  total = c(orig, synt)

  bins = seq(min(total), max(total), length.out = nbins)

  bins[1] = -Inf

  b_total = as.numeric(cut(total, breaks=bins))

  b_orig = b_total[seq(length(orig))]
  b_synt = b_total[seq(length(orig)+1,length(b_total))]

  orig_prob = c()
  synt_prob = c()

  for (level in sort(unique(b_total))){

    orig_prob = c(orig_prob, sum(b_orig == level) / length(b_orig))
    synt_prob = c(synt_prob, sum(b_synt == level) / length(b_synt))

  }

  probs  = rbind(orig_prob, synt_prob)

  return(probs)

}

column_prob_disc = function(orig, synt){

  total = c(orig,synt)

  orig_prob = c()
  synt_prob = c()

  for (level in sort(unique(total))){

    orig_prob = c(orig_prob, sum(orig == level) / length(orig))
    synt_prob = c(synt_prob, sum(synt == level) / length(synt))

  }

  probs = rbind(orig_prob, synt_prob)

  return(probs)

}

kl_divergence = function(p,q){

  return(sum(ifelse((p != 0) & (q != 0), p * log(p / q), 0)))

}

js_divergence = function(p,q){

  m <- (p+q)/2

  return(kl_divergence(p, m)/2 + kl_divergence(q, m))

}

wasserstein_distance_2 = function(X,Y){

  max_ <- max(max(X), max(Y))
  min_ <- min(min(X), min(Y))

  norm_X <- (X - min_) / (max_ - min_)
  norm_Y <- (Y - min_) / (max_ - min_)

  return((sum((abs(sort(norm_X) - sort(norm_Y)))^2) / length(X))^(1/2))

}

# compute KLD and WD

# original data and synthetic data are dataframes.

# col_info is a list containing index of columns which are continuous or discrete

# here, discrete does not mean categorical variables, but discrete random variables that are in numerical form such as counts

# ex) col_info[['cont']] = c(1,2,4), col_info[['cate']] = c(0,3)


distance_measures = function(original_data, synthetic_data, col_info){

  # get prob for each column

  col_prob = list()

  for (col in col_info[['cont']]){
    col_prob[[col]] = column_prob_cont(original_data[,col], synthetic_data[,col])
  }

  for (col in col_info[['disc']]){
    col_prob[[col]] = column_prob_disc(original_data[,col], synthetic_data[,col])
  }

  # get KL-divergence for each column

  col_kld = list()

  kld_data = 0

  for (i in seq(ncol(original_data))){

    kld = kl_divergence(col_prob[[i]][1,], col_prob[[i]][2,])

    col_kld[[i]] = kld

    kld_data = kld_data + kld

  }

  # get js-divergence for each column

  col_jsd = list()

  jsd_data = 0

  for (i in seq(ncol(original_data))){

    jsd = js_divergence(col_prob[[i]][1,], col_prob[[i]][2,])

    col_jsd[[i]] = jsd

    jsd_data = jsd_data + jsd

  }

  # get wasserstein distance for each column

  col_wd = list()

  wd_data = 0

  for (col in colnames(original_data)){

    wd = wasserstein_distance_2(original_data[[col]], synthetic_data[[col]])

    col_wd[[col]] = wd

    wd_data = wd_data + wd

  }

  return(list('jsd_data' = jsd_data, 'kld_data' =  kld_data, 'wd_data' = wd_data,
              'col_jsd' = col_jsd, 'col_kld' = col_kld, 'col_wd' = col_wd)) # return KLD, WD of data and column-wise KLD, WD.


}

##################################################
# network.R
##################################################

# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


# -----------------------------------------
# Construction of feature representations
# -----------------------------------------
#
# + build_network:
#   --------------
#           |
#           +--------> feedforward_network:
#           |
#           +--------> recurrent_network:
#           |
#           +--------> MNIST_network:

# TODO: add arguments details

# Global variables
ACTIVATION_DICT <- list(ReLU = torch::nn_relu,
                        Hardtanh = torch::nn_hardtanh,
                        ReLU6 = torch::nn_relu6,
                        Sigmoid = torch::nn_sigmoid,
                        Tanh = torch::nn_tanh,
                        ELU = torch::nn_elu,
                        CELU = torch::nn_celu,
                        SELU = torch::nn_selu,
                        GLU = torch::nn_glu,
                        LeakyReLU = torch::nn_leaky_relu,
                        LogSigmoid = torch::nn_log_sigmoid,
                        Softplus = torch::nn_softplus
)

build_network <- function(network_name, params) {
  if (network_name == "feedforward") {
    net <- feedforward_network(params)
  }
  return(net)
}

feedforward_network <- function(params) {
  # Architecture for a Feedforward Neural Network
  #
  #  Args:
  #
  #      ::params::
  #
  #      ::params["input_dim"]::
  #      ::params[""rep_dim""]::
  #      ::params["num_hidden"]::
  #      ::params["activation"]::
  #      ::params["num_layers"]::
  #      ::params["dropout_prob"]::
  #      ::params["dropout_active"]::
  #      ::params["LossFn"]::
  #
  #  Returns:
  #
  #      ::_architecture::

  modules <- list()

  if (params$dropout_active) {
    modules <- append(modules, torch::nn_dropout(p=params$dropout_prob))
  }

  # Input layer

  modules <- append(modules, torch::nn_linear(in_features=params$input_dim, out_features=params$num_hidden, bias=FALSE))
  modules <- append(modules, ACTIVATION_DICT[[params$activation]]())

  # Intermediate layers

  for (u in 1:(params$num_layers-1)) {
    if (params$dropout_active) {
      modules <- append(modules, torch::nn_dropout(p=params$dropout_prob))
    }
    modules <- append(modules, torch::nn_linear(in_features=params$num_hidden, out_features=params$num_hidden, bias=FALSE))
    modules <- append(modules, ACTIVATION_DICT[[params$activation]]())
  }

  # Output layer

  modules <- append(modules, torch::nn_linear(in_features=params$num_hidden, out_features=params$rep_dim, bias=FALSE))

  architecture <- torch::nn_sequential()
  for (i in 1:length(modules)) {
    architecture$add_module(name = i, module = modules[[i]])
  }

  return(architecture)
}

##################################################
# OneClass.R
##################################################

# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# One-class loss functions
# ------------------------

OneClassLoss <- function(outputs, c) {
  dist   <- torch::torch_sum((outputs - c) ** 2, dim=2)
  loss   <- torch::torch_mean(dist)

  return(loss)
}

SoftBoundaryLoss <- function(outputs, R, c, nu) {
  dist   <- torch::torch_sum((outputs - c) ** 2, dim=2)
  scores <- dist - R ** 2
  loss   <- R ** 2 + (1 / nu) * torch::torch_mean(torch::torch_maximum(torch::torch_zeros_like(scores), scores))

  scores <- dist
  loss   <- (1 / nu) * torch::torch_mean(torch::torch_maximum(torch::torch_zeros_like(scores), scores))

  return(loss)
}

LossFns <- list(OneClass = OneClassLoss, SoftBoundary = SoftBoundaryLoss)

# Base network
# ---------------------

BaseNet <- R6::R6Class(classname = "BaseNet",
                   inherit = torch::nn_module(),
                   public = list(
                     initialize = function() {

                       super$initialize()

                       self$logger <- logging::getLogger(name = deparse(self$classname))
                       self$rep_dim <- NULL # representation dimensionality, i.e. dim of the last layer

                     },
                     forward = function(...) {

                       stop("NotImplementedError")

                     },
                     summary = function() {
                       net_parameters <- list()
                       params <- 0

                       for (p in self$parameters){
                         if (p$requires_grad){
                           net_parameters <- append(net_parameters, p)
                         }
                       }

                       for (p in net_parameters){
                         params <- sum(params + prod(p$size()))
                       }

                       loginfo(sprintf("Trainable parameters: %s", params))
                       loginfo(deparse(self$classname))
                     }
                   )
)

get_radius <- function(dist, nu) {
  if(!any(sapply(class(dist), function(x) x == 'torch_tensor'))){
    stop("Input dist argument must be torch_tensor")
  }
  if(!is.double(nu)){
    stop("Input nu argument must be double")
  }

  radius <- quantile(sqrt(as.numeric(dist)), 1 - nu, na.rm = T)
  return(radius)
}

# Define OneClassLayer class
OneClassLayer <- R6::R6Class(classname = "OneClassLayer",
                         inherit = BaseNet,
                         lock_objects = FALSE,

                         public = list(

                           initialize = function(params = NULL, hyperparams = NULL) {

                             super$initialize()

                             # set all representation parameters - remove these lines

                             self$rep_dim <- params$rep_dim
                             self$input_dim <- params$input_dim
                             self$num_layers <- params$num_layers
                             self$num_hidden <- params$num_hidden
                             self$activation <- params$activation
                             self$dropout_prob <- params$dropout_prob
                             self$dropout_active <- params$dropout_active
                             self$loss_type <- params$LossFn
                             self$train_prop <- params$train_prop
                             self$learningRate <- params$lr
                             self$epochs <- params$epochs
                             self$warm_up_epochs <- params$warm_up_epochs
                             self$weight_decay <- params$weight_decay

                             self$device <- torch::torch_device(ifelse(torch::cuda_is_available(), "cuda", "cpu")) # Make this an option

                             # Set up the network

                             self$model <- build_network(network_name = "feedforward", params = params)
                             self$model$to(device = self$device)

                             # Create the loss function

                             self$c <- hyperparams$center$to(device = self$device)
                             self$R <- hyperparams$Radius
                             self$nu <- hyperparams$nu

                             self$loss_fn <- LossFns[[self$loss_type]]
                           },

                           forward = function(x) {
                             x <- self$model(x)
                             return(x)
                           },

                           fit = function(x_train, verbosity = TRUE) {
                             self$optimizer <- torch::optim_adamw(self$model$parameters, lr = self$learningRate, weight_decay = self$weight_decay)

                             self$X <- torch::torch_tensor(matrix(x_train, ncol=self$input_dim), dtype = torch::torch_float32())

                             if (self$train_prop != 1) {
                               train_len <- floor(self$train_prop * dim(x_train)[1])
                               x_val <- x_train[(train_len+1):dim(x_train)[1], ]
                               x_train <- x_train[1:train_len, ]
                               inputs_val <- torch::torch_tensor(x_val, requires_grad = TRUE)$to(dtype = torch::torch_float32(),
                                                                                          device = self$device)
                             }

                             self$losses <- list()
                             self$loss_vals <- list()

                             for (epoch in 1:self$epochs){

                               # Converting inputs and labels to Variable

                               inputs <- torch::torch_tensor(x_train, requires_grad = TRUE)$to(dtype = torch::torch_float32(),
                                                                                        device = self$device)

                               self$model$zero_grad()
                               self$optimizer$zero_grad()

                               # get output from the model, given the inputs
                               outputs <- self$model(input = inputs)

                               # get loss for the predicted output

                               if (self$loss_type == "SoftBoundary") {
                                 self$loss <- self$loss_fn(outputs = outputs, R = self$R, c = self$c, nu = self$nu)
                               } else if (self$loss_type == "OneClass") {
                                 self$loss <- self$loss_fn(outputs = outputs, c = self$c)
                               }

                               # get gradients w.r.t to parameters
                               self$loss$backward(retain_graph = TRUE)
                               self$losses <- append(self$losses, as.matrix(self$loss$detach()))

                               # update parameters
                               self$optimizer$step()

                               if (epoch >= self$warm_up_epochs && self$loss_type == "SoftBoundary") {
                                 dist <- torch::torch_sum((outputs - self$c)^2, dim = 1)
                                 # self$R <- torch$tensor(get_radius(dist, self$nu))
                               }

                               if (self$train_prop != 1.0) {
                                 torch::with_no_grad({

                                   # get output from the model, given the inputs
                                   outputs <- self$model(inputs_val)

                                   # get loss for the predicted output

                                   if (self$loss_type == "SoftBoundary") {
                                     loss_val <- self$loss_fn(outputs = outputs, R = self$R, c = self$c, nu = self$nu)
                                   } else if (self$loss_type == "OneClass") {
                                     loss_val <- self$loss_fn(outputs = outputs, c = self$c)
                                   }

                                   self$loss_vals <- append(self$loss_vals, loss_val)
                                 })
                               }

                               if (verbosity) {
                                 if (self$train_prop == 1) {
                                   cat('epoch {', epoch, '}, loss {', as.matrix(self$loss), '}\n')

                                 } else {
                                   cat('epoch {', sprintf("%.4f", epoch),
                                       '}, train loss {', sprintf("%.4e", as.matrix(self$loss)),
                                       '}, val loss {', sprintf("%.4e", as.matrix(loss_val)), '}\n')
                                 }
                               }
                             }
                             return(self)
                           })
)

##################################################
# evaluation.R
##################################################

# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel
# Licensed under the BSD 3-clause license (see LICENSE.txt)

#  -----------------------------------------
#  Metrics implementation
#  -----------------------------------------


compute_alpha_precision <- function(real_data, synthetic_data, emb_center, compute_authen = TRUE) {

  emb_center <- torch::torch_tensor(emb_center, device=device)

  n_steps <- 30
  nn_size <- 2
  alphas <- seq(0, 1, length.out = n_steps)

  Radii <- as.matrix(torch::torch_tensor(quantile(as.matrix(torch::torch_sqrt(torch::torch_sum((torch::torch_tensor(real_data, dtype = torch::torch_float32()) - emb_center)^2, dim = -1))), alphas)))

  alpha_precision_curve <- numeric()

  synth_to_center <- torch::torch_sqrt(torch::torch_sum((torch::torch_tensor(synthetic_data, dtype = torch::torch_float32()) - emb_center)^2, dim = -1))

  for (k in 1:length(Radii)) {
    precision_audit_mask <- as.numeric(synth_to_center <= Radii[k])
    alpha_precision <- mean(precision_audit_mask)

    alpha_precision_curve <- c(alpha_precision_curve, alpha_precision)
  }

  Delta_precision_alpha <- 1 - 2 * sum(abs(alphas - alpha_precision_curve)) * (alphas[2] - alphas[1])

  authenticity = NULL

  if (compute_authen == TRUE){

    nbrs_real <- FNN::knn(real_data, real_data, k=2, cl=1:nrow(real_data))
    real_to_real <- attributes(nbrs_real)$nn.dist

    nbrs_synth <- FNN::knn(real_data, synthetic_data, k=1, cl=1:nrow(real_data))
    real_to_synth <- attributes(nbrs_synth)$nn.dist
    real_to_synth_args <- attributes(nbrs_synth)$nn.index

    # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real <- torch::torch_tensor(real_to_real[,2])
    real_to_synth <- torch::torch_tensor(real_to_synth)
    real_to_synth_args <- torch::torch_tensor(real_to_synth_args)


    # See which one is bigger
    authen <- real_to_real[real_to_synth_args] < real_to_synth
    authenticity <- mean(as.numeric(authen))
  }

  return(list(alphas = alphas, alpha_precision_curve = alpha_precision_curve, Delta_precision_alpha = Delta_precision_alpha, authenticity = authenticity))
}

##################################################
# pra_measures.R
##################################################

compute_metrics <- function(real_data, synthetic_data, seed = 42, rep_dim = NULL, compute_authen = FALSE) {

  set.seed(seed)

  X <- as.matrix(real_data)
  Y <- as.matrix(synthetic_data)

  results = list()

  if (is.null(rep_dim)) {
    rep_dim <- ncol(X)
  }

  params <- list(
    rep_dim = rep_dim,
    num_layers = 2,
    num_hidden = 200,
    activation = "ReLU",
    dropout_prob = 0.5,
    dropout_active = FALSE,
    train_prop = 1,
    epochs = 100,
    warm_up_epochs = 10,
    lr = 1e-3,
    weight_decay = 1e-2,
    LossFn = "SoftBoundary"
  )

  hyperparams <- list(Radius = 1, nu = 1e-2)

  params$input_dim <- ncol(X)
  hyperparams$center <- torch::torch_ones(params$rep_dim)

  # embedding of real_data

  model_real <- OneClassLayer$new(params = params, hyperparams = hyperparams)

  model_real$fit(X, verbosity = FALSE)

  X_out_real <- as.matrix(torch::with_no_grad(torch::torch_tensor(model_real$forward(torch::torch_tensor(X, dtype = torch::torch_float32())), dtype = torch::torch_float32())))
  Y_out_real <- as.matrix(torch::with_no_grad(torch::torch_tensor(model_real$forward(torch::torch_tensor(Y, dtype = torch::torch_float32())), dtype = torch::torch_float32())))

  metrics_real <- compute_alpha_precision(X_out_real, Y_out_real, model_real$c)
  alphas <- metrics_real$alphas

  # embedding of synthetic_data
  model_synth <- OneClassLayer$new(params = params, hyperparams = hyperparams)

  model_synth$fit(Y, verbosity = FALSE)

  X_out_synth <- as.matrix(torch::with_no_grad(torch::torch_tensor(model_synth$forward(torch::torch_tensor(X, dtype = torch::torch_float32())), dtype = torch::torch_float32())))
  Y_out_synth <- as.matrix(torch::with_no_grad(torch::torch_tensor(model_synth$forward(torch::torch_tensor(Y, dtype = torch::torch_float32())), dtype = torch::torch_float32())))

  metrics_synth <- compute_alpha_precision(Y_out_synth, X_out_synth, model_synth$c, compute_authen = FALSE)

  results <- list(
    Dpa = metrics_real$Delta_precision_alpha,
    apc = metrics_real$alpha_precision_curve,
    Dcb = metrics_synth$Delta_precision_alpha,
    bcc = metrics_synth$alpha_precision_curve,
    mean_aut = metrics_real$authenticity
  )

  return(list(results, model_real, model_synth))
}


plot_pr_curve <- function(alpha_precision_curve, beta_coverage_curve, authenticity) {
  alphas <- seq(0, 1, length.out = 30)

  plot(alphas, alpha_precision_curve, type = "l", col = "blue", lwd = 2, main = paste("alpha-Precision and beta-Recall curve\n(Authenticity =", round(authenticity, 3), ")"))
  lines(alphas, beta_coverage_curve, col = "purple", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "red")
  legend("topleft", legend = c("alpha-precision", "beta-recall"), col = c("blue", "purple"), lwd = 2, lty = 1)
}


pra_sampling <- function(real_data, synthetic_data, n_sampling, seed = 42, pair = FALSE) {

  set.seed(seed)

  iap <- numeric()
  ibr <- numeric()
  aut <- numeric()

  apc <- numeric()
  brc <- numeric()


  for (i in 1:n_sampling) {

    if (pair) {

      sample_index <- sample(seq_len(nrow(real_data)), 5000, replace = FALSE)

      sample_real <- real_data[sample_index, ]
      sample_synth <- synthetic_data[sample_index, ]

    } else {
      real_data <- real_data[sample(seq_len(nrow(real_data))), ]
      synthetic_data <- synthetic_data[sample(seq_len(nrow(synthetic_data))), ]

      sample_real <- real_data[1:5000, ]
      sample_synth <- synthetic_data[1:5000, ]
    }

    results <- compute_metrics(sample_real, sample_synth)

    iap <- c(iap, results[[1]]$Dpa)
    ibr <- c(ibr, results[[1]]$Dcb)
    aut <- c(aut, results[[1]]$mean_aut)
    apc <- c(apc, results[[1]]$apc)
    brc <- c(brc, results[[1]]$bcc)
  }

  return(list(iap = iap, ibr = ibr, aut = aut, apc = apc, brc = brc))
}


##################################################
# pMSE_R.R
##################################################

###-----utility.gen--------------------------------------------------------
utility.gen2 <- function(object, data, ...) UseMethod("utility.gen2")


###-----utility.gen.default------------------------------------------------
utility.gen2.default <- function(object, ...)
  stop("No compare method associated with class ", class(object), call. = FALSE)


###-----utility.gen.data.frame---utility.gen.list--------------------------
utility.gen2.data.frame <- utility.gen2.list <-
  function(object, data,
           not.synthesised = NULL, cont.na = NULL,
           method = "cart", maxorder = 1,
           k.syn = FALSE, tree.method = "rpart",
           max.params = 400, print.stats = c("pMSE", "S_pMSE","standard_pMSE"),
           resamp.method = NULL, nperms = 50, cp = 1e-3,
           minbucket = 5, mincriterion = 0, vars = NULL,
           aggregate = FALSE, maxit = 200, ngroups = NULL,
           print.flag = TRUE, print.every = 10,
           digits = 6, print.zscores = FALSE, zthresh = 1.6,
           print.ind.results = FALSE,
           print.variable.importance = FALSE, ...)
  {

    if (is.null(data)) stop("Requires parameter 'data' to give name of the real data.\n\n",  call. = FALSE)
    if (is.null(object)) stop("Requires parameter 'object' to give name of the synthetic data.\n\n",  call. = FALSE)

    if (is.list(object) & !is.data.frame(object)) m <- length(object)
    else if (is.data.frame(object)) m <- 1
    else stop("object must be a data frame or a list of data frames.\n", call. = FALSE)

    # sort out cont.na to make it into a complete named list
    cna <- cont.na
    cont.na <- as.list(rep(NA, length(data)))
    names(cont.na) <- names(data)
    if (!is.null(cna)) {
      if (!is.list(cna) | any(names(cna) == "") | is.null(names(cna)))
        stop("Argument 'cont.na' must be a named list with names of selected variables.", call. = FALSE)
      if (any(!names(cna) %in% names(data))) stop("Names of the list cont.na must be variables in data.\n", call. = FALSE)
      for (i in 1:length(cna)) {
        j <- (1:length(data))[names(cna)[i] == names(data)]
        cont.na[[j]] <- unique(c(NA,cna[[i]]))
      }
    }

    syn.method = rep("ok", length(data))
    if (!is.null(not.synthesised)) {
      if (!is.null(not.synthesised) && !all(not.synthesised %in% names(data))) stop("not.synthesised must be names of variables in data.\n", call. = FALSE)
      syn.method[names(data) %in% not.synthesised] <- ""
    }

    object <- list(syn = object, m = m, strata.syn = NULL, method = syn.method, cont.na = cont.na)
    class(object ) <- "synds"

    res <- utility.gen2.synds(object = object, data = data,
                              method = method, maxorder = maxorder,
                              k.syn = k.syn, tree.method = tree.method,
                              max.params = max.params, print.stats = print.stats,
                              resamp.method = resamp.method, nperms = nperms, cp = cp,
                              minbucket = minbucket, mincriterion = mincriterion,
                              vars = vars, aggregate = aggregate, maxit = maxit,
                              ngroups = ngroups, print.flag = print.flag,
                              print.every = print.every, digits = digits,
                              print.zscores = print.zscores, zthresh = zthresh,
                              print.ind.results = print.ind.results,
                              print.variable.importance = print.variable.importance)
    res$call <- match.call()
    return(res)
  }


###-----utility.gen--------------------------------------------------------
utility.gen2.synds <- function(object, data,
                               method = "cart", maxorder = 1,
                               k.syn = FALSE, tree.method = "rpart",
                               max.params = 400, print.stats = c("pMSE", "S_pMSE","standard_pMSE"),
                               resamp.method = NULL, nperms = 50, cp = 1e-3,
                               minbucket = 5, mincriterion = 0, vars = NULL,
                               aggregate = FALSE, maxit = 200, ngroups = NULL,
                               print.flag = TRUE, print.every = 10,
                               digits = 6, print.zscores = FALSE,
                               zthresh = 1.6, print.ind.results = FALSE,
                               print.variable.importance = FALSE, ...)
{
  m  <- object$m

  # Check input parameters
  if (is.null(method) || length(method) != 1 || is.na(match(method, c("cart", "logit"))))
    stop("Invalid 'method' type - must be either 'logit' or 'cart'.\n", call. = FALSE)
  if (is.null(print.stats) || any(is.na(match(print.stats, c("pMSE", "SPECKS", "PO50", "U", "S_pMSE","standard_pMSE", "S_SPECKS", "S_PO50", "S_U", "all")))))
    stop("Invalid 'print.stats'. Can only include 'pMSE', 'SPECKS', 'PO50', 'U', 'S_pMSE','standard_pMSE', 'S_SPECKS', 'S_PO50', 'S_U'.\nAternatively it can be set to 'all'.\n", call. = FALSE)
  if (!is.null(resamp.method) && is.na(match(resamp.method, c("perm", "pairs", "none"))))
    stop("Invalid 'resamp.method' type - must be NULL, 'perm', 'pairs' or 'none'.\n", call. = FALSE)
  if (aggregate == TRUE & method != "logit") stop("Aggregation only works for 'logit' method.\n", call. = FALSE)
  if (is.null(data)) stop("Requires parameter 'data' to give name of the real data.\n",  call. = FALSE)
  if (!inherits(object, "synds")) stop("Object must have class 'synds'.\n", call. = FALSE)
  if (k.syn & !is.null(resamp.method) && resamp.method == "pairs") stop('\nresamp.method = "pairs" will give the wrong answer when k.syn is TRUE.\n', call. = FALSE)
  if (is.null(tree.method) || length(tree.method) != 1 || is.na(match(tree.method, c("rpart", "ctree"))))
    stop("Invalid 'tree.method' - must be either 'rpart' or 'ctree'.\n", call. = FALSE)


  # Check selected variables and make observed and synthetic comparable
  if (!(is.null(vars))) {
    if (is.numeric(vars)){
      if (!(all(vars %in% 1:length(data)))) stop("Column indices of 'vars' must be in 1 to length(data).\n", call. = FALSE)
    } else if (!(all(vars %in% names(data)))) stop("Some 'vars' specified not in data.\n", call. = FALSE)
    data <- data[, vars, drop = FALSE]
    if (m == 1) {
      if (!all(vars %in% names(object$syn))) stop("Some 'vars' specified not in synthetic data.\n", call. = FALSE)
      else object$syn <- object$syn[, vars, drop = FALSE ]
    } else {
      if (!all(vars %in% names(object$syn[[1]]))) stop("Some 'vars' specified not in synthetic data.\n", call. = FALSE)
      else object$syn <- lapply(object$syn, "[", vars)
    }
  } else {
    if (m == 1) vars <- names(object$syn) else vars <- names(object$syn[[1]])
    if (!all(vars %in% names(data))) stop("Some variables in synthetic data not in original data.\n", call. = FALSE)
    else data <- data[, vars]  # make data match synthetic
  }

  # get cont.na and method parameters for stratified synthesis
  if (!is.null(object$strata.syn)) {
    cna <- object$cont.na[1,]
    syn.method <- object$method[1,]
  } else {
    cna <- object$cont.na
    syn.method <- object$method
  }

  cna <- cna[names(cna) %in% vars]

  for ( i in 1:length(cna)) {
    nm <- names(cna)[i]
    vals <- unique(cna[[i]][!is.na(cna[[i]])])  # get variables with cont.na other than missing
    if (length(vals) > 0){
      for (j in 1:length(vals))
        n_cna <- sum(vals[j] == data[,nm] & !is.na(data[,nm]))
      if (n_cna == 0) stop("\nValue ", vals[j], " identified as denoting a special or missing in cont.na for ",nm, " is not in data.\n",sep = "", call. = FALSE)
      else if (n_cna < 10 & print.flag) cat ("\nWarning: Only ",n_cna ," record(s) in data with value ",vals[j]," identified as denoting a missing value in cont.na for ",nm, "\n\n", sep = "")
    }
  }
  # Check whether some variables are unsynthesised
  incomplete <- FALSE
  nsynthd <- length(vars)
  unsyn.vars <- names(syn.method)[syn.method == ""]  # identify unsynthesised
  if (any(vars %in% unsyn.vars) & !is.null(unsyn.vars)) {
    notunsyn <- vars[!vars %in% unsyn.vars]  # synthesised vars
    if (!all(unsyn.vars %in% vars)) stop("Unsynthesised variables must be a subset of variables contributing to the utility measure.\n", call. = FALSE)
    if ( all(vars %in% unsyn.vars)) stop("Utility measure impossible if all in vars are unsynthesised.\n", call. = FALSE)
    incomplete <- TRUE
  }

  # Set default resampling according to completeness and print.stats (incl. S_SPECKS or S_PO50 or S_U)
  if (is.null(resamp.method)) {
    if ("S_SPECKS" %in% print.stats || "S_PO50" %in% print.stats || "S_U" %in% print.stats || incomplete) {
      resamp.method <- "pairs"
      cat('Resampling method set to "pairs" because S_SPECKS or S_PO50 or S_U in print.stats or incomplete = TRUE.\n')
    } else if (method == "cart") resamp.method <- "perm"
  } else {
    if (incomplete & resamp.method == "perm")
      stop('Incomplete synthesis requires resamp.method = "pairs".\n', call. = FALSE)
    if (any(c("S_SPECKS", "S_PO50", "S_U") %in% print.stats) & resamp.method == "perm")
      stop('Stat SPECKS, PO50, and U requires resamp.method = "pairs" to get S_SPECKS, S_PO50, and S_U respectively.\n', call. = FALSE)
    if (resamp.method == "pairs" & m == 1)
      stop('resamp.method = "pairs" needs a synthesis with m > 1, m = 10 suggested.\n', call. = FALSE)
  }

  # Drop any single value columns
  leneq1 <- function(x) length(table(as.numeric(x[!is.na(x)]), useNA = "ifany")) %in% (0:1)

  dchar <- sapply(data,is.character)
  if (any(dchar == TRUE)) for ( i in 1:dim(data)[2]) if (dchar[i] == TRUE) data[,i] <- factor(data[,i])
  dout <- sapply(data,leneq1)
  if (m == 1) sout <- sapply(object$syn,leneq1)
  else  sout <- sapply(object$syn[[1]],leneq1)
  dout <- dout & sout
  if (any(dout == TRUE) & print.flag) {
    cat("Some columns with single values or all missing values in original and synthetic\nexcluded from utility comparisons (excluded variables: ",
        paste(names(data)[dout], collapse = ", "), ").\n", sep = "")
    data <- data[,!dout]
    if (m == 1) object$syn <- object$syn[, !dout, drop = FALSE]
    else object$syn <- lapply(object$syn, "[", !dout)
  }

  # Numeric variables
  numvars <- (1:dim(data)[2])[sapply(data, is.numeric)]
  names(numvars) <- names(data)[numvars]
  # If ngroups != NULL divide numeric variables into ngroups
  data0 <- data  # to save if m > 1

  if (!is.null(ngroups)) {
    for (i in numvars) {
      if (m == 1) {
        groups <- group_num(data[,i], object$syn[,i], object$syn[,i],
                            ngroups, cont.na = cna, ...)
        data[,i] <- groups[[1]]
        object$syn[,i] <- groups[[2]]
      } else {
        syn0 <- c(sapply(object$syn, '[[', i))
        for (j in 1:m) {
          groups <- group_num(data0[,i], object$syn[[j]][,i], syn0,
                              ngroups, cont.na = cna[[i]], ...)
          data[,i] <- groups[[1]]
          object$syn[[j]][,i] <- groups[[2]]
        }
      }
    }
  }

  # Categorical vars: make missing data part of factor
  catvars <- (1:dim(data)[2])[sapply(data, is.factor)]
  for (i in catvars) {
    data[,i] <- factor(data[,i])
    if (m == 1) object$syn[,i] <- factor(object$syn[,i])
    else for (j in 1:m) object$syn[[j]][,i] <- factor(object$syn[[j]][,i])
    if (any(is.na(data[,i]))) {
      data[,i] <- addNA(data[,i])
      if (m == 1) object$syn[,i] <- addNA(object$syn[,i])
      else for (j in 1:m) object$syn[[j]][,i] <- addNA(object$syn[[j]][,i])
    }
  }

  for (i in numvars) {
    if (anyNA(data[,i]) & is.null(ngroups)) {
      newname <- paste(names(data)[i], "NA", sep = "_")
      data <- data.frame(data, 1*(is.na(data[,i])))
      names(data)[length(data)] <- newname
      data[is.na(data[,i]), i] <- 0
      if (m == 1) {
        object$syn <- data.frame(object$syn, 1*(is.na(object$syn[,i])))
        names(object$syn)[length(object$syn)] <- newname
        object$syn[is.na(object$syn[,i]), i] <- 0
      } else {
        for (j in 1:m) {
          object$syn[[j]] <- data.frame(object$syn[[j]], 1*(is.na(object$syn[[j]][,i])))
          names(object$syn[[j]])[length(object$syn[[j]])] <- newname
          object$syn[[j]][is.na(object$syn[[j]][,i]),i] <- 0
        }
      }
    }
    if (any(!is.na(cna[[i]]))  & is.null(ngroups)) {
      cna[[i]] <- cna[[i]][!is.na(cna[[i]])]
      for (j in 1:length(cna[[i]])) {
        newname <- paste(names(data)[i], "cna",j, sep = "_")
        data <- data.frame(data, 1*(data[,i] == cna[[i]][j]))
        data[data[,i] == cna[[i]][j], i] <- 0
        names(data)[length(data)] <- newname
      }
      if (m == 1) {
        for (j in 1:length(cna[[i]])) {
          newname <- paste(names(object$syn)[i], "cna",j, sep = "_")
          object$syn <- data.frame(object$syn, 1*(object$syn[,i] == cna[[i]][j]))
          object$syn[object$syn[,i] == cna[[i]][j], i] <- 0
          names(object$syn)[length(object$syn)] <- newname
        }
      } else {
        for (k in 1:m) {
          for (j in 1:length(cna[[i]])) {
            newname <- paste(names(object$syn[[k]])[i], "cna",j, sep = "_")
            object$syn[[k]] <- data.frame(object$syn[[k]], 1*(object$syn[[k]][,i] == cna[[i]][j]))
            object$syn[[k]][object$syn[[k]][,i] == cna[[i]][j], i] <- 0
            names(object$syn[[k]])[length(object$syn[[k]])] <- newname
          }
        }
      }
    }
  }

  # Function for getting propensity scores
  # --------------------------------------
  propcalcs <- function(syndata, data) {

    n1 <- dim(data)[1]
    n2 <- dim(syndata)[1]
    N <- n1 + n2
    cc <- n2 / N
    if (k.syn) cc <- 0.5

    df.prop <- rbind(syndata, data)  # make data frame for calculating propensity score
    df.prop <- data.frame(df.prop, t = c(rep(1,n2), rep(0,n1)))

    # remove any levels of factors that don't exist in data or syndata
    catvars <- (1:(dim(df.prop)[2]))[sapply(df.prop,is.factor)]
    for (i in catvars) {
      if (any(table(df.prop[,i]) == 0)) {
        df.prop[,i] <- as.factor(as.character(df.prop[,i]))
        if (print.flag) cat("Empty levels of factor(s) for variable ", names(df.prop)[i]," removed.\n" )
      }
    }

    if (aggregate == TRUE) {
      aggdat <- aggregate(df.prop[,1], by = df.prop, FUN = length)
      wt <- aggdat$x
      aggdat <- aggdat[, -dim(aggdat)[2]]
    }

    if (method == "logit" ) {

      if (maxorder >= dim(data)[2])
        stop("maxorder cannot be greater or equal to the number of variables.\n", call. = FALSE)

      # cheking for large models
      levs <- sapply(data, function(x) length(levels(x)))
      levs[levs == 0] <- 2
      tt1 <- apply(combn(length(levs), 1), 2, function(x) {prod(levs[x] - 1)})
      if (maxorder == 0) nparams <- 1 + sum(tt1)
      else {
        tt2 <- apply(combn(length(levs), 2), 2, function(x) {prod(levs[x] - 1)})
        if (maxorder == 1) nparams <- 1 + sum(tt1) + sum(tt2)
        else {
          tt3 <- apply(combn(length(levs), 3), 2, function(x) {prod(levs[x] - 1)})
          if (maxorder == 2) nparams <- 1 + sum(tt1) + sum(tt2) + sum(tt3)
          else {
            tt4 <- apply(combn(length(levs), 4), 2, function(x) {prod(levs[x] - 1)})
            if (maxorder == 3) nparams <- 1 +  sum(tt1) + sum(tt2) + sum(tt3) + sum(tt4)
            else {
              tt5 <- apply(combn(length(levs), 5), 2, function(x) {prod(levs[x] - 1)})
              if (maxorder == 4) nparams <- 1 +  sum(tt1) + sum(tt2) + sum(tt3) + sum(tt4) + sum(tt5)
            }
          }
        }
      }
      if (nparams > max.params) stop("You will be fitting a large model with ", nparams,
                                     " parameters that may take a long time and fail to converge.
Have you selected variables with vars?
You can try again, if you really want to, by increasing max.params.\n", sep = "", call. = FALSE)
      else if (nparams > dim(data)[[1]]/5) cat("You will be fitting a large model with ", nparams,
                                               " parameters and only ", dim(data)[[1]], " records
that may take a long time and fail to converge.
Have you selected variables with vars?\n")

      if (maxorder >= 1) logit.int <- as.formula(paste("t ~ .^", maxorder + 1))
      else logit.int <- as.formula(paste("t ~ ."))

      if (aggregate == TRUE) fit <- glm(logit.int, data = aggdat, family = "binomial",
                                        control = list(maxit = maxit), weights = wt)
      else fit <- suppressWarnings(glm(logit.int, data = df.prop, family = "binomial",
                                       control = list(maxit = maxit)))
      #if (fit$converged == FALSE) cat("\nConvergence failed.\n")

      # Get number of parameters that involve synthesised variables
      score <- predict(fit, type = "response")
      if (incomplete == FALSE) km1 <- length(fit$coefficients[!is.na(fit$coefficients)]) - 1  # To allow for non-identified coefficients
      else {
        namescoef <- names(fit$coefficients)
        coefOK <- rep(FALSE, length(namescoef))
        for (nn in notunsyn) coefOK[grepl(nn, namescoef)] <- TRUE
        km1 <- sum(coefOK & print.flag)
        if (m == 1 || (m > 1 & j == 1)) cat("Expectation of utility uses only coefficients involving synthesised variables: ",
                                            km1, " from ", length(fit$coefficients) - 1, "\n", sep = "")
      }
      # one more coefficient (intercept needed if k.syn TRUE)
      if (k.syn) km1 <- km1 + 1
      if (aggregate == TRUE) {
        pMSE <- (sum(wt*(score - cc)^2, na.rm = T)) / N
        KSt <- suppressWarnings(ks.test(rep(score[aggdat$t == 1], wt[aggdat$t == 1]),
                                        rep(score[aggdat$t == 0], wt[aggdat$t == 0])))
        SPECKS <- KSt$statistic
        PO50 <- sum(wt[(score > 0.5 & df.prop$t == 1) | ( score <= 0.5 & df.prop$t == 0)])/N*100 - 50
        U      <- suppressWarnings(wilcox.test(rep(score[aggdat$t == 1], wt[aggdat$t == 1]),
                                               rep(score[aggdat$t == 0], wt[aggdat$t == 0]))$statistic)
      } else {
        pMSE <- (sum((score - cc)^2, na.rm = T)) / N
        KSt <- suppressWarnings(ks.test(score[df.prop$t == 1], score[df.prop$t == 0]))
        SPECKS <- KSt$statistic
        PO50 <- sum((score > 0.5 & df.prop$t == 1) | ( score <= 0.5 & df.prop$t == 0))/N*100 - 50
        U      <- suppressWarnings(wilcox.test(score[df.prop$t == 1], score[df.prop$t == 0])$statistic)
      }
      pMSEExp <- km1 * (1 - cc)^2 * cc / N
      pMSEsd <- sqrt(2*(km1-1))* (1 - cc)^2 * cc / N
      S_pMSE  <- pMSE / pMSEExp
      standard_pMSE<- (pMSE-pMSEExp)/pMSEsd

      # to save space
      fit$data <- NULL
      # fit$model <- fit$residuals <- fit$y <- NULL ?

    } else if (method == "cart") {
      km1 <- NA
      if (tree.method == "rpart") {
        fit <- rpart::rpart(t ~ ., data = df.prop, method = 'class',
                     control = rpart::rpart.control(cp = cp, minbucket = minbucket))
        score <- predict(fit)[, 2]
      } else if (tree.method == "ctree") {
        fit <- ctree(t ~ ., data = df.prop,
                     controls = ctree_control(mincriterion = mincriterion, minbucket = minbucket))
        score <- predict(fit)
      }
      pMSE <- sum((score - cc)^2, na.rm = T) / N
      KSt <- suppressWarnings(ks.test(score[df.prop$t == 1], score[df.prop$t == 0]))
      SPECKS <- KSt$statistic
      PO50 <- sum((score > 0.5 & df.prop$t == 1) | ( score <= 0.5 & df.prop$t == 0))/N*100 - 50
      U <- suppressWarnings(wilcox.test(score[df.prop$t == 1], score[df.prop$t == 0])$statistic)
    }

    # Permutations
    if (!is.null(resamp.method) && resamp.method == "none") S_pMSE <- standard_pMSE <- NA
    else if (!is.null(resamp.method) && resamp.method == "perm") { # to allow resamp for logit models
      S_pMSE<- standard_pMSE <- rep(NA, m)
      simpMSE <- rep(0, nperms)
      if (m == 1) j <- 1
      if (j == 1 & print.flag) {
        if (print.every == 0 | print.every >= nperms) cat("Running ", nperms, " permutations to get NULL utilities.", sep = "")
        else cat("Running ", nperms, " permutations to get NULL utilities and printing every ", print.every, "th.", sep = "")
      }
      #if (print.flag) cat("\nsynthesis ", j, "   ", sep = "")
      if (print.flag) cat("\nsynthesis ")

      for (i in 1:nperms) {
        if (print.every > 0 & nperms > print.every & floor(i/print.every) == i/print.every & print.flag)  cat(i, " ", sep = "")
        pdata <- df.prop
        if (!k.syn) pdata$t <- sample(pdata$t)
        else pdata$t <- rbinom(N, 1, 0.5)

        if (method == "cart") {
          if (tree.method == "rpart") {
            sfit <- rpart::rpart(t ~ ., data = pdata, method = 'class', control = rpart::rpart.control(cp = cp, minbucket = minbucket))
            score <- predict(sfit)[,2]
          } else if (tree.method == "ctree") {
            sfit <- ctree(t ~ ., data = pdata,
                          controls = ctree_control(mincriterion = mincriterion, minbucket = minbucket))
            score <- predict(sfit)
          }
          simpMSE[i] <- (sum((score - cc)^2, na.rm = T)) / N / 2

        } else if (method == "logit") {
          if (maxorder >= 1) logit.int <- as.formula(paste("t ~ .^", maxorder + 1))
          else logit.int <- as.formula(paste("t ~ ."))

          if (aggregate == TRUE) {
            aggdat1 <- aggregate(pdata[,1], by = pdata, FUN = length)
            wt <- aggdat1$x
            aggdat1 <- aggdat1[, -dim(aggdat1)[2]]
            sfit <- glm(logit.int, data = aggdat1, family = "binomial",
                        control = list(maxit = maxit), weights = wt)
          } else sfit <- glm(logit.int, data = pdata, family = "binomial",
                             control = list(maxit = maxit))

          if (sfit$converged == FALSE & print.flag) cat("Warning: Logistic model did not converge in ",
                                                        maxit, " iterations.\nYou could try increasing parameter 'maxit'.\n", sep = "")
          score <- predict(sfit, type = "response")
          if (aggregate == TRUE) {
            simpMSE[i] <- sum(wt*(score - cc)^2, na.rm = T) / N / 2 # reduced by factor of 2
          } else {
            simpMSE[i] <- sum((score - cc)^2, na.rm = T) / N / 2 # reduced by factor of 2
          }
        }
      }
      nnosplits <- c(sum(simpMSE < 1e-8), length(simpMSE))
      S_pMSE <- pMSE/mean(simpMSE)
      standard_pMSE <-(pMSE-mean(simpMSE))/sd(simpMSE)
    }
    if (!is.null(resamp.method) && resamp.method == "pairs")
      res.ind <- list(pMSE = pMSE, SPECKS = SPECKS, PO50 = PO50, U = U,
                      S_pMSE= NA,standard_pMSE=NA, S_SPECKS = NA,  S_PO50 = NA, S_U = NA,
                      fit = fit, nnosplits = NA, df = NA)
    else if (!is.null(resamp.method) && resamp.method == "perm")
      res.ind <- list(pMSE = pMSE, SPECKS = SPECKS, PO50 = PO50,U = U,
                      S_pMSE= S_pMSE, standard_pMSE=standard_pMSE, S_SPECKS = NA, S_PO50 = NA, S_U = NA,
                      fit = fit, nnosplits = nnosplits, df = NA)
    else res.ind <- list(pMSE = pMSE, SPECKS = SPECKS, PO50 = PO50, U =U,
                         S_pMSE = S_pMSE,standard_pMSE=standard_pMSE, S_SPECKS = NA, S_PO50 = NA, S_U = NA,
                         fit = fit, nnosplits = NA, df = km1) ## changed to NA
    return(res.ind)
  }
  # --------------------------------------
  # end propcalcs

  n1 <- nrow(data)

  if (m == 1) {
    n2 <- nrow(object$syn)
    res.ind <- propcalcs(object$syn, data)
    res <- list(call = match.call(), m = m, method = method, tree.method = tree.method,
                resamp.method = resamp.method, maxorder = maxorder, vars = vars,
                k.syn = k.syn, aggregate = aggregate, maxit = maxit,
                ngroups = ngroups, mincriterion = mincriterion,
                nperms = nperms, df = res.ind$df, incomplete = incomplete,
                pMSE = res.ind$pMSE, S_pMSE = res.ind$S_pMSE, standard_pMSE = res.ind$standard_pMSE,
                S_SPECKS = res.ind$S_SPECKS, S_PO50 = res.ind$S_PO50,S_U = res.ind$S_U,
                SPECKS = res.ind$SPECKS, PO50 = res.ind$PO50, U = res.ind$U,
                print.stats = print.stats,
                fit = res.ind$fit, nnosplits = res.ind$nnosplits,
                digits = digits, print.ind.results = print.ind.results,
                print.zscores = print.zscores, zthresh = zthresh,
                print.variable.importance = print.variable.importance)
  } else {
    n2 <- nrow(object$syn[[1]])
    pMSE <- SPECKS <- PO50 <- U <- S_pMSE<-standard_pMSE <- S_SPECKS <- S_PO50 <- S_U <- rep(NA, m)
    fit <- nnosplits <- as.list(1:m)
    if (!is.null(resamp.method) && !(resamp.method == "none") && resamp.method == "pairs") {
      kk <- 0
      simpMSE <- simKS <- simPO50 <- simU <- rep(NA, m*(m - 1)/2)
    }
    for (j in 1:m) {
      res.ind <- propcalcs(object$syn[[j]], data)
      pMSE[j] <- res.ind$pMSE
      SPECKS[j] <- res.ind$SPECKS
      PO50[j] <- res.ind$PO50
      U[j] <- res.ind$U
      fit[[j]] <- res.ind$fit

      if (resamp.method == "none" || (method == "logit" & (is.null(resamp.method)))) {
        if (j == 1 & print.flag) cat("Fitting syntheses: ")
        if (print.flag) {
          cat(j, " ", sep = "")
          if (res.ind$fit$converged == FALSE) cat("Convergence failed.\n")
        }
        if (j == m ) cat("\n")
        S_pMSE[j] <- res.ind$S_pMSE
        standard_pMSE[j]<-res.ind$standard_pMSE
      }

      if (!is.null(resamp.method) && resamp.method == "pairs") {
        if (j == 1 & print.flag) {
          if (print.every == 0 | m*(m - 1)/2 <= print.every) cat("Simulating NULL pMSE from ", m*(m - 1)/2, " pair(s).", sep = "")
          else cat("Simulating NULL pMSE from ", m*(m - 1)/2, " pairs, printing every ", print.every, "th:\n", sep = "")
          if (m*(m - 1)/2 < 6 ) cat("\nNumber of pairs too low, we suggest increasing number of syntheses (m).\n")
        }
        if (j < m) {
          for (jj in (j + 1):(m)) {
            kk <- kk + 1
            if (print.every > 0 & print.every < m*(m - 1)/2 & floor(kk/print.every) == kk/print.every & print.flag) cat(kk," ",sep = "")
            simvals <- propcalcs(object$syn[[j]], object$syn[[jj]])
            simpMSE[kk] <- simvals$pMSE
            simKS[kk] <- simvals$SPECKS
            simPO50[kk] <- simvals$SPECKS
            simU[kk] <- simvals$U
          }
        }
        nnosplits<- c(sum(simpMSE < 1e-8), length(simpMSE))
        for (j in 1:m) {
          S_pMSE[j] <- pMSE[j] *2 /mean(simpMSE)
          standard_pMSE[j]<-(pMSE[j]-mean(simpMSE))/sd(simpMSE)
          S_SPECKS[j] <- SPECKS[j] *2 /mean(simKS)
          S_PO50[j] <- PO50[j] *2 /mean(simPO50)
          S_U[j] <- U[j] *2 /mean(simU)
        }

      } else {
        nnosplits[[j]] <- res.ind$nnosplits
        S_pMSE[j] <- res.ind$S_pMSE
        standard_pMSE[j]<-  res.ind$standard_pMSE
        S_SPECKS[j] <- res.ind$S_SPECKS
        S_PO50[j] <- res.ind$S_PO50
        S_U[j] <- res.ind$S_U
      }
    }
    res <- list(call = match.call(), m = m, method = method, tree.method = tree.method,
                resamp.method = resamp.method, maxorder = maxorder, vars = vars,
                k.syn = k.syn, aggregate = aggregate, maxit = maxit,
                ngroups = ngroups, mincriterion = mincriterion,
                nperms = nperms, df = res.ind$df, incomplete = incomplete,
                pMSE = pMSE,  S_pMSE = S_pMSE,standard_pMSE=standard_pMSE,
                S_SPECKS = S_SPECKS, S_PO50 = S_PO50, S_U = S_U,
                SPECKS = SPECKS, PO50 = PO50, U = U,
                print.stats = print.stats,
                fit = fit, nnosplits = nnosplits,
                digits = digits, print.ind.results = print.ind.results,
                print.zscores = print.zscores, zthresh = zthresh,
                print.variable.importance = print.variable.importance)

  }
  class(res) <- "utility2.gen"
  res$call <- match.call()
  return(res)
}





###-----group_num----------------------------------------------------------
# function to categorise continuous variables

group_num <- function(x1, x2, xsyn, n = 5, style = "quantile", cont.na = NA, ...) {

  # Categorise 2 continuous variables into factors of n groups
  # with same groupings determined by the first one
  # xsyn - all synthetic values (for m syntheses)

  if (!is.numeric(x1) | !is.numeric(x2) | !is.numeric(xsyn))
    stop("x1, x2, and xsyn must be numeric.\n", call. = FALSE)

  # Select non-missing(nm) values
  x1nm <- x1[!(x1 %in% cont.na) & !is.na(x1)]
  x2nm <- x2[!(x2 %in% cont.na) & !is.na(x2)]
  xsynnm <- xsyn[!(xsyn %in% cont.na) & !is.na(xsyn)]

  # Derive breaks
  my_breaks <- unique(suppressWarnings(classIntervals(c(x1nm, xsynnm),
                                                      n = n, style = style, ...))$brks)

  my_levels <- c(levels(cut(x1nm, breaks = my_breaks,
                            dig.lab = 8, right = FALSE, include.lowest = TRUE)),
                 cont.na[!is.na(cont.na)])

  # Apply groupings to non-missing data
  x1[!(x1 %in% cont.na) & !is.na(x1)] <- as.character(cut(x1nm,
                                                          breaks = my_breaks, dig.lab = 8, right = FALSE, include.lowest = TRUE))
  x2[!(x2 %in% cont.na) & !is.na(x2)] <- as.character(cut(x2nm,
                                                          breaks = my_breaks, dig.lab = 8, right = FALSE, include.lowest = TRUE))
  x1 <- factor(x1, levels = my_levels)
  x2 <- factor(x2, levels = my_levels)

  return(list(x1,x2))
}


##################################################
# Confidence interval overlap.R
##################################################

# original_data : original dataset
# synthetic_data : synthetic dataset
# target : response variable name
# Measure : population.inference = FALSE(default) or TRUE
# m : imputation count
# If m>1, the synthetic dataset must be arranged in a row.
# For example, if m=5 then synthetic_data = cbind(synthetic_data_1, ..., synthetic_data_5).

CISynthpop = function(original_data, synthetic_data, target, Measure, m){
  synthetic_data_ = synthetic_data
  q = matrix(rep(NA,ncol(original_data)*m), nrow=m)
  std = matrix(rep(NA,ncol(original_data)*m), nrow=m)
  k = ncol(synthetic_data)/m
  n = nrow(original_data)

  ##for original data

  #Fitting GLM ft for original data
  form = as.formula(paste0(target, '~.', sep=""))
  results = glm(form, family=gaussian, data=original_data)

  #coefficient, standard error, and 95% confidence interval for original data
  original_coef = results$coefficients
  original_std = sqrt(diag(vcov(results)))
  original_CI = confint(profile(fitted = results), level = 0.95)

  ##for synthetic data
  for (i in 1:m){
    synthetic_data = synthetic_data_[,(k*(i-1)+1):(k*i)]
    p = ncol(synthetic_data)

    #Fitting GLM ft for synthetic data
    form = as.formula(paste0(target, '~.', sep=""))
    results = glm(form, family=gaussian, data=synthetic_data)

    #coefficient, standard error, and 95% confidence interval for synthetic data
    synthetic_coef = results$coefficients
    synthetic_std = sqrt(diag(vcov(results)))
    synthetic_CI = confint(profile(fitted = results), level = 0.95)

    #Append coefficient and std of synthetic data
    q[i,] = synthetic_coef
    std[i,] = synthetic_std
  }

  #Calculate coefficient and std of synthetic data
  synthetic_coef = colMeans(q)
  synthetic_std = colMeans(std)

  #Calculate value of IO measure when population.inference = FALSE
  if (Measure == "FALSE"){
    value_list = 1-abs((synthetic_coef-original_coef)/original_std)/(2*1.96)
    result = mean(value_list, na.rm=TRUE)
  }

  if (Measure == "True"){
    k = nrow(synthetic_data)
    synthetic_std_ = synthetic_std * sqrt(1/m + k/n)

    p = nrow(original_CI)
    value_list = rep(NA, p)
    for (i in 1:p){
      synthetic_CIl = synthetic_coef[i]-1.96*synthetic_std_[i]
      synthetic_CIu = synthetic_coef[i]+1.96*synthetic_std_[i]
      overlap_lower = max(original_CI[i], synthetic_CIl)
      overlap_upper = min(original_CI[p+i], synthetic_CIu)
      value = 0.5 * (((overlap_upper - overlap_lower) / (original_CI[p+i] - original_CI[i]))
                     + ((overlap_upper - overlap_lower) / (synthetic_CIu - synthetic_CIl)))
      value_list[i] = value
    }
    result = sum(value_list, na.rm=TRUE) / length(value_list[!is.na(value_list)])
  }
  return(result)
}

##################################################
# Identity disclosure risk.R
##################################################

# orig : original dataset
# syn : synthetic dataset
# sensitive_var : list of sensitive variables names
# sensitive_conti_k : list that returns k if the sensitive variable is continuous and 0 if it is categorical.

Disclosure = function(orig, syn, sensitive_var, sensitive_conti_k){

  KMEANS = function(orig, syn, sensitive_var, sensitive_conti_k){
    # categorization
    data_cat_orig <- orig
    data_cat_syn <- syn
    data_clu_or <- orig

    #K-means -> categorization
    # K-means(original data) -> cluster -> categorization of original_data and synthetic_data
    Km <- kmeans(data.frame(orig[sensitive_var]), centers = sensitive_conti_k)
    data_clu_or[sensitive_var] <- Km$cluster
    clu_d <- data.frame(cbind(data_clu_or[sensitive_var], orig[sensitive_var]))
    colnames(clu_d) <- c("clu","value")
    clu_d <- arrange(dplyr::summarize(dplyr::group_by(clu_d, clu), value = max(value)), value)
    clu_value <- clu_d$value

    # categorization
    sensitive_vark = paste0(sensitive_var , "_k")
    data_cat_orig[sensitive_vark] <- 1
    data_cat_syn[sensitive_vark] <- 1
    for(n in 1:(length(clu_value)-1)){
      data_cat_orig[sensitive_vark] <- data_cat_orig[sensitive_vark] + (orig[sensitive_var] > clu_value[n]) * 1
    }

    return(data_cat_orig)
  }

  var_name = colnames(orig)
  quasi_var = var_name[-which(var_name %in% sensitive_var)]

  # number of quasi-identifiers
  n_quasi = length(quasi_var)

  # number of sensitive variables
  n_sensitive = length(sensitive_var)
  n_conti = 0; n_nominal = 0

  for (i in sensitive_conti_k){
    if (i != 0){
      n_conti = n_conti + 1
    }else{
      n_nominal = n_nominal + 1
    }
  }

  if (n_conti != 0){
    for (i in 1:n_sensitive){
      if (sensitive_conti_k[i] != 0){
        orig = KMEANS(orig, syn, sensitive_var[i], sensitive_conti_k[i])
      }
    }
  }

  result = 0

  for (i in 1:nrow(orig)){

    for (j in 1:n_quasi){
      v1 = paste0("value", j)
      v2 = orig[i, quasi_var[j]]
      assign(v1, v2)
    }

    df1 = orig
    # calculate f
    for (j in 1:n_quasi){
      v1 = eval(parse(text = paste0("value", j)))
      df1 = df1[which(as.character(df1[,quasi_var[j]]) == as.character(v1)),]
    }

    f = nrow(df1)

    # measuring identification risk (calculate I)
    df2 = syn
    for (j in 1:n_quasi){
      v1 = eval(parse(text = paste0("value", j)))
      df2 = df2[which(as.character(df2[,quasi_var[j]]) == as.character(v1)),]
    }

    num = nrow(df2)
    index2 = rownames(df2)

    if (num != 0){
      I = 1
    }else if (num == 0){
      I = 0
    }

    # learning something new (calculate R)
    if (I == 0){
      R = 0
    }else if (I != 0){
      L = 0

      for (j in 1:n_sensitive){
        var = sensitive_var[j]

        if (sensitive_conti_k[j] == 0){
          sensitive_value = sort(unique(orig[,var]))
        }else if (sensitive_conti_k[j] != 0){
          vark = paste0(var, "_k")
          sensitive_value = sort(unique(orig[,vark]))
        }

        p_j_list = c(); d_j_list = c()
        for (k in sensitive_value){
          if (sensitive_conti_k[j] == 0){
            prop = length(which(orig[var] == k)) / nrow(orig)
            p_j_list = c(p_j_list, prop)
            d_j_list = c(d_j_list, 1-prop)
          }else if (sensitive_conti_k[j] != 0){
            prop = length(which(orig[vark] == k)) / nrow(orig)
            p_j_list = c(p_j_list, prop)
            d_j_list = p_j_list
          }
        }

        # choose random among matched synthetic records
        index2 = sample(index2, 1)
        Y_t = syn[index2, var]

        if (sensitive_conti_k[j] == 0){
          X_s = orig[i, var]
        }else if (sensitive_conti_k[j] != 0){
          X_s = orig[i, vark]
        }
        index = which(sensitive_value == X_s)

        # for nominal variable
        if (sensitive_conti_k[j] == 0){

          if (X_s == Y_t){
            value = 1
          }else if (X_s != Y_t){
            value = 0
          }

          p_j = p_j_list[index]; d_j = d_j_list[index]
          value1 = d_j * value
          value2 = sqrt(p_j * (1-p_j))

          if (value1 > value2){
            L = L + 1
          }else if (value1 <= value2){
            L = L
          }
        }else if (sensitive_conti_k[j] != 0){
          # for continuous variable

          d_j = d_j_list[index]
          value = abs(X_s - Y_t)
          x = orig[,var]
          MAD = median(abs(x-median(x)))

          value1 = d_j * value
          value2 = 1.48*MAD

          if (value1 < value2){
            L = L + 1
          }else if (value1 >= value2){
            L = L
          }
        }
      }

      if (L >= 0.05*n_sensitive){
        R = 1
      }else if (L < 0.05*n_sensitive){
        R = 0
      }
    }

    # final calculation
    final = 1/f * I * R
    result = result + final

  }

  result = result / nrow(orig)
  return (result)
}

##################################################
# CAP_attribute_risk.R
##################################################

CAP <- function(original_data, synthetic_data, attribute,conti_var=FALSE, K_list=FALSE){

  # categorization
  data_orig <- original_data
  data_syn <- synthetic_data[, colnames(data_orig)]
  if (sum(conti_var!=FALSE)==0){
    df_or <- data_orig
    df_syn <- data_syn
  }else{
    data_cat_orig <- data_orig
    data_cat_syn <- data_syn
    data_clu_or <- data_orig

    #K-means -> categorization
    for(i in 1:length(conti_var)){
      # K-means(original data) -> cluster -> categorization of original_data and synthetic_data
      Km <- kmeans(data.frame(data_orig[conti_var[i]]), centers = K_list[i])
      data_clu_or[conti_var[i]] <- Km$cluster
      clu_d <- data.frame(cbind(data_clu_or[conti_var[i]], data_orig[conti_var[i]]))
      colnames(clu_d) <- c("clu","value")
      clu_d <- dplyr::arrange(dplyr::summarize(dplyr::group_by(clu_d, clu), value = max(value)), value)
      clu_value <- clu_d$value

      # categorization
      data_cat_orig[conti_var[i]] <- 1
      data_cat_syn[conti_var[i]] <- 1
      for(n in 1:(length(clu_value)-1)){
        data_cat_orig[conti_var[i]] <- data_cat_orig[conti_var[i]] + (data_orig[conti_var[i]] > clu_value[n]) * 1
        data_cat_syn[conti_var[i]] <- data_cat_syn[conti_var[i]] + (data_syn[conti_var[i]] > clu_value[n]) * 1
      }
    }
    df_or <- data_cat_orig
    df_syn <- data_cat_syn
  }


  ###########################################################################################
  # CAP
  X_or <- df_or[, !(colnames(df_or) %in% attribute)]
  X_or.n <- data.frame(matrix(nrow = nrow(X_or), ncol = ncol(X_or)))
  X_or.n[,] <- lapply(X_or[,],as.numeric)
  X_syn <- df_syn[, !(colnames(df_syn) %in% attribute)]
  X_syn.n <- data.frame(matrix(nrow = nrow(X_syn), ncol = ncol(X_syn)))
  X_syn.n[,] <- lapply(X_syn[,],as.numeric)
  # list of unique values for each variable
  unique_X_or=unique(X_or)

  # CAP
  na_count <- 0
  p_sum <- 0
  for(i in 1:nrow(unique_X_or)){
    target_list <- unique_X_or[i,]
    target_matrix_or=matrix(rep(as.numeric(target_list),nrow(X_or)),nrow(X_or),by=T)
    target_matrix_syn=matrix(rep(as.numeric(target_list),nrow(X_syn)),nrow(X_syn),by=T)

    index_target_or=as.matrix(abs(X_or.n-target_matrix_or))%*%matrix(1,ncol(X_or),1)==0
    index_target_syn=as.matrix(abs(X_syn.n-target_matrix_syn))%*%matrix(1,ncol(X_or),1)==0
    target_or <- df_or[index_target_or,]
    target_syn <- df_syn[index_target_syn,]
    if(nrow(target_syn) == 0){
      na_count <- na_count + nrow(target_or)
    }else{
      attribute_matrix=matrix(0,2,length(unique(target_or[[attribute]])))
      for(t in 1:length(unique(target_or[[attribute]]))){
        uni_t=unique(target_or[[attribute]])[t]
        attribute_matrix[1,t]=nrow(target_or[as.character(target_or[[attribute]]) == as.character(uni_t),])
        attribute_matrix[2,t]=nrow(target_syn[as.character(target_syn[[attribute]]) == as.character(uni_t),])/nrow(target_syn)
      }
      p_sum <- p_sum + sum(attribute_matrix[1,]*attribute_matrix[2,])
      na_count <- na_count + sum((attribute_matrix[1,])*(attribute_matrix[2,]==0))
    }
  }

  CAP_0 <- p_sum / nrow(data_orig) # CAP_0 (consider case not in original as 0)
  CAP_NA <- p_sum / (nrow(data_orig) - na_count) # CAP_NA (do not consider case not in original)
  
  return(list(CAP_0=CAP_0, CAP_NA=CAP_NA))
}

##################################################
# Population_uniqueness.R
##################################################

pop_uni <- function(original_data, synthetic_data, conti_var=c(), K_list=FALSE){
  data_orig <- original_data
  data_syn <- synthetic_data[, names(original_data)]

  # categorization
  if (length(conti_var) == 0) {
    df_or <- data_orig
    df_syn <- data_syn
  } else {
    data_cat_orig <- data_orig
    data_cat_syn <- data_syn
    data_clu_or <- data_orig

    # K-means -> categorization
    for (i in seq_along(conti_var)) {
      # K-means(original data) -> cluster -> categorization of original_data and synthetic_data
      Km <- kmeans(as.matrix(data_orig[[conti_var[i]]]), centers = as.integer(K_list[i]))
      data_clu_or[[conti_var[i]]] <- as.integer(Km$cluster)
      clu_d <- data.frame(clu = data_clu_or[[conti_var[i]]], value = data_orig[[conti_var[i]]])
      clu_d <- as.data.frame(dplyr::arrange(dplyr::summarise(dplyr::group_by(clu_d, clu), value=max(value)), value))
      clu_value <- clu_d$value

      # categorization
      data_cat_orig[[conti_var[i]]] <- 1
      data_cat_syn[[conti_var[i]]] <- 1
      for (n in seq_along(clu_value)[-length(clu_value)]) {
        data_cat_orig[[conti_var[i]]] <- data_cat_orig[[conti_var[i]]] + (data_orig[[conti_var[i]]] > clu_value[n]) * 1
        data_cat_syn[[conti_var[i]]] <- data_cat_syn[[conti_var[i]]] + (data_syn[[conti_var[i]]] > clu_value[n]) * 1
      }
    }

    df_or <- data_cat_orig
    df_syn <- data_cat_syn
  }

  # Population uniqueness
  no_dup_X_syn <- unique(df_syn)
  count_syn <- vector()
  for (i in seq_len(nrow(no_dup_X_syn))) {
    syn_i <- df_syn[rowSums(df_syn == as.list(no_dup_X_syn[i, ]), na.rm=TRUE) == ncol(no_dup_X_syn), ]
    count_syn <- c(count_syn, nrow(syn_i))
  }
  uni_X_syn <- no_dup_X_syn[count_syn == 1, ]

  # unique case of original data among unique synthetic data
  count_or <- vector()
  for (i in seq_len(nrow(uni_X_syn))) {
    count_orig_i <- sum(rowSums(df_or == as.list(uni_X_syn[i, ]), na.rm=TRUE) == ncol(uni_X_syn))
    count_or <- c(count_or, count_orig_i)
  }

  population_uni <- sum(count_or == 1) / nrow(uni_X_syn)

  return(population_uni)
}

##################################################
# Record Linkage.R
##################################################

record_linkage <- function(x, x_link, block, scale){

  exp_sim = function(x, origin=0, offset=0, sc=1){
    tmp1 <- abs(x-origin) - offset
    tmp1[tmp1<0] <- 0
    tmp2 = 2^(-tmp1/sc)
    return(tmp2)
  }  # similarity function

  comp_column <- names(x)[!names(x) %in% block]

  # index pairs to be compared : ind.tmp
  D1 = as.data.frame(x[, block])
  D2 = as.data.frame(x_link[, block])

  ind1 = unique(D1)
  L1 = list(nrow(ind1))
  L2 = list(nrow(ind1))

  for (i in 1:nrow(ind1)){
    ind1_df = data.frame(matrix(nrow=nrow(D1), ncol=ncol(D1)))
    ind1_df[,] = ind1[i,]
    L1[[i]] = which(rowSums(D1 == ind1_df)==length(block))
    L2[[i]] = which(rowSums(D2 == ind1_df)==length(block))
  }

  Lmin = apply(cbind(unlist(lapply(L1, length)), unlist(lapply(L2,length))),1,min)

  candidate_links <- foreach::`%dopar%`(foreach::foreach(i = which(Lmin>0), .combine = rbind),  {
    expand.grid(L1[[i]], L2[[i]])
  })

  # compute similarity function (link_sc --> link_diff)
  link_D = x[candidate_links[,1],comp_column] - x_link[candidate_links[,2],comp_column]
  link_sc = link_D
  link_sc <- foreach::`%dopar%`(foreach::foreach(i = 1:length(comp_column), .combine = cbind), {
    exp_sim(link_D[i], sc = scale[i], offset = 0, origin = 0)
  })

  features = cbind(candidate_links, link_sc)
  features$sum = rowSums(link_sc, na.rm = T)

  return(features)
}

##################################################
# mia_measures_kde.R
##################################################

compute_mia <- function(original_data, synthetic_data, threshold = 2) {

  bw <- sqrt(diag(ks::Hns(x = as.matrix(original_data))))
  bw[bw == 0] <- 0.1

  orig_dens_est <- ks::kde(x = as.matrix(original_data), H=diag(bw), eval.points = as.matrix(original_data), binned = FALSE)
  p_R <- orig_dens_est$estimate
  syn_dens_est <- ks::kde(x = as.matrix(synthetic_data), H=diag(bw), eval.points = as.matrix(original_data), binned = FALSE)
  p_S <- syn_dens_est$estimate

  prop <- sum((p_S / p_R) > threshold)/nrow(original_data)

  return(list(p_S = p_S, p_R = p_R, prop = prop))

}

mia_sampling <- function(original_data, synthetic_data, threshold = 2, n_sampling = 10, seed = 42, pair = FALSE) {

  set.seed(seed)

  mia <- list()
  for (i in 1:n_sampling) {

    if (pair == TRUE) {

      sample_index <- sample(1:nrow(original_data), 5000, replace = FALSE)

      sample_orig <- original_data[sample_index, ]
      sample_syn <- synthetic_data[sample_index, ]

    } else {

      sample_orig <- original_data[sample(1:nrow(original_data), 5000, replace = FALSE), ]
      sample_syn <- synthetic_data[sample(1:nrow(synthetic_data), 5000, replace = FALSE), ]

    }

    result <- compute_mia(sample_orig, sample_syn, threshold)
    mia[[i]] <- result

  }

  return(mia)

}

##################################################
# DCR_NNDR.R
##################################################

DCR_NNDR <- function(original_data, synthetic_data, col_info){
  m <- nrow(synthetic_data)

  # compute continuous scaler

  orig_cont <- original_data[, col_info$cont]
  synt_cont <- synthetic_data[, col_info$cont]

  max_min <- (apply(orig_cont, 2, max) - apply(synt_cont, 2, min))^2
  min_max <- (apply(orig_cont, 2, min) - apply(synt_cont, 2, max))^2

  cont_scale <- apply(cbind(max_min, min_max), 1, max)


  # compute ordinal scaler

  orig_ordi <- original_data[, col_info$ordi]
  synt_ordi <- synthetic_data[, col_info$ordi]

  max_min <- abs(apply(orig_ordi, 2, max) - apply(synt_ordi, 2, min))
  min_max <- abs(apply(orig_ordi, 2, min) - apply(synt_ordi, 2, max))

  ordi_scale <- apply(cbind(max_min, min_max), 1, max)


  # compute distance for each synthetic data

  orig_cate <- original_data[, col_info$cate]
  synt_cate <- synthetic_data[, col_info$cate]

  # New
  cont_scale_df = matrix(cont_scale, nrow=nrow(data.frame(synt_cont)), ncol=length(cont_scale), byrow=T)
  ordi_scale_df = matrix(ordi_scale, nrow=nrow(data.frame(synt_cont)), ncol=length(ordi_scale), byrow=T)

  dist_mat <- matrix(0, nrow = m, ncol = 2)

  for (i in 1:m) {
    # New
    synt_cont_df = matrix(as.numeric(as.data.frame(synt_cont)[i,]), nrow=nrow(data.frame(synt_cont)), ncol=ncol(data.frame(synt_cont)), byrow=T)
    synt_ordi_df = matrix(as.numeric(as.data.frame(synt_ordi)[i,]), nrow=nrow(data.frame(synt_ordi)), ncol=ncol(data.frame(synt_ordi)), byrow=T)
    synt_cate_df = matrix(as.numeric(as.data.frame(synt_cate)[i,]), nrow=nrow(data.frame(synt_cate)), ncol=ncol(data.frame(synt_cate)), byrow=T)

    # conti distance (squared Euclidean)
    cont_dist <- apply((as.matrix(orig_cont)-synt_cont_df)^2 / cont_scale_df, 1, sum)
    ordi_dist <- apply(abs(as.matrix(orig_ordi)-synt_ordi_df) / ordi_scale_df, 1, sum)
    cate_dist <- apply(as.matrix(orig_cate) == synt_cate_df, 1, sum)

    total_dist <- sort(cont_dist + ordi_dist + cate_dist)

    dist_mat[i, ] <- total_dist[c(1, 2)]
  }


  DCR <- dist_mat[, 1]
  NNDR <- dist_mat[, 1] / dist_mat[, 2]

  return(list(DCR=DCR, NNDR=NNDR))
}

#' Compute a Utility and Risk Measure
#'
#'
#' This function calculates utility and risk measures for synthetic data.
#' x, x_link, and measure are required parameters,
#' and required parameters exist depending on the measure (see example)
#'
#' @param x data.frame; original data
#' @param x_link data.frame; synthetic data
#' @param measure character; utility and risk measures \{ "w_distance", "kl_divergence", "pmse", "standard_pmse", "alpha_precision", "beta_recall", "cl_overlap", "idr", "cap", "pop_uni", "record_linkage", "mia", "dcr", "nndr", "authen" \}
#' @param col_info list of column position information; \{ "cont": continuous, "cate": category, "ordi": ordinary (dcr, nndr only)\}; using measure: "w_distance", "kl_divergence", "js_divergence", "dcr", "nndr" 
#' @param resamp.method character vector of resampling method, input NULL when not using this function; \{ "perm", "pairs", "none" \}
#' @param sensitive_var character vector of sensitive variables names(= block column); using measure: "idr", "cap", "record_linkage"
#' @param sensitive_conti_k column position information list of sensitive variables; using measure: "idr"
#' @param conti_var character vector of continuous variables; There is information that overlaps with col_info, but there is a function that receives parameters in that format. using measure: "cap", "pop_uni" 
#' @param K_list integer vector of center k values to be used for clustering; using measure: "cap", "pop_uni" 
#' @param scale numeric vector of record linkage scale by column; using measure: "record_linkage" 
#' @param thres integer of record linkage threshold; using measure: "record_linkage"
#' @param iteration integer of record linkage iteration; Number of iterations to use in random sampling of record linkage; default=1, using measure: "record_linkage"
#' @param n_sampling integer of record linkage number of random sampling, input FALSE when not using this function; default=FALSE, using measure: "record_linkage" 
#' @param cl_target character; target variable used for cl_overlap; using measure: "cl_overlap"
#' @param cl_measure character or logical; \{ 'True', 'FALSE', TRUE, FALSE \}; using measure: "cl_overlap"
#' @param cl_m integer; using measure: "cl_overlap"
#' @return A numeric or numeric list
#' @examples 
#' # ## Utility
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'w_distance')
#' print(paste('w_distance:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'kl_divergence')
#' print(paste('kl_divergence:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'js_divergence')
#' print(paste('js_divergence:', result))
#' result <- Compute_Measure(orig_fact_df, syn_fact_df, measure = 'pmse')
#' print(paste('pmse:', result))
#' result <- Compute_Measure(orig_fact_df, syn_fact_df, measure = 'standard_pmse')
#' print(paste('standard_pmse:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'alpha_precision')
#' print(paste('alpha_precision:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'beta_recall')
#' print(paste('beta_recall:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'cl_overlap', cl_target='BIS_MNTH')
#' print(paste('cl_overlap:', result)) # >> glm의 coefficient가 NA를 뱉는 경우 NA를 반환하게 작성되어 있음.
#' ## Risk
#' result <- Compute_Measure(orig_fact_df, syn_fact_df, measure = 'idr', sensitive_var = 'BIS_MNTH')
#' print(paste('idr:', result))
#' ## library(MASS)
#' result_list <- Compute_Measure(orig_fact_df, syn_fact_df, measure = 'cap', sensitive_var = 'BIS_MNTH')# needs MASS to be installed
#' print(paste('cap:', result_list)) # >> CAP_0, CAP_NA로 이루어진 리스트 반환
#' result <- Compute_Measure(orig_fact_df, syn_fact_df, measure = 'pop_uni')
#' print(paste('pop_uni:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'record_linkage', sensitive_var = 'BIS_MNTH')
#' print(paste('record_linkage:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'mia')
#' print(paste('mia:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'dcr')$DCR_mean
#' print(paste('dcr:', result))
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'nndr')$NNDR_mean
#' print(paste('nndr:', result)) # >> DCR, NNDR이 list형태로 반환되며 각각 nrow의 길이를 갖고 있어 각각의 평균 및 전체값을 리스트 형태로 반환 
#' result <- Compute_Measure(orig_num_df, syn_num_df, measure = 'authen')
#' print(paste('authen:', result))
#' @export
Compute_Measure <- function(x, x_link, measure, col_info=NULL, resamp.method='perm',
                            sensitive_var=NULL, sensitive_conti_k=FALSE, conti_var=FALSE, K_list=FALSE,
                            scale=NULL, thres=NULL, iteration=1, n_sampling=FALSE,
                            cl_target=NULL, cl_measure='True', cl_m=1){

  # orig : original dataset
  # syn : synthetic dataset
  # sensitive_var : list of sensitive variables names
  # sensitive_conti_k : list that returns k if the sensitive variable is continuous and 0 if it is categorical.
  quasi_var <- setdiff(colnames(x), sensitive_var)

  ##################################################
  # wasserstein_distance
  ##################################################
  if (measure == 'w_distance'){

    if (is.null(col_info)) {
      col_info <- list(cont = which(sapply(x, is.numeric)),
                       cate = which(!sapply(x, is.numeric)))
      # col_info is a list containing index of columns which are continuous or discrete
      # here, discrete does not mean categorical variables, but discrete random variables that are in numerical form such as counts
      # ex) col_info[['cont']] = c(1,2,4), col_info[['cate']] = c(3,5)
    }

    result <- distance_measures(original_data = x, synthetic_data = x_link, col_info = col_info)$wd_data

    ##################################################
    # KL_divergence
    ##################################################
  }else if (measure == 'kl_divergence'){

    if (is.null(col_info)) {
      col_info <- list(cont = which(sapply(x, is.numeric)),
                       cate = which(!sapply(x, is.numeric)))
    }

    result <- distance_measures(original_data = x, synthetic_data = x_link, col_info = col_info)$kld_data
    ##################################################
    # js divergence
    ##################################################
  }else if (measure == 'js_divergence'){

    if (is.null(col_info)) {
      col_info <- list(cont = which(sapply(x, is.numeric)),
                       disc = which(!sapply(x, is.numeric)))
    }

    result <- distance_measures(original_data = x, synthetic_data = x_link, col_info = col_info)$jsd_data

    ##################################################
    # pMSE
    ##################################################
  }else if (measure == 'pmse'){

    result <- utility.gen2(x, x_link)$pMSE

    ##################################################
    # standardized pMSE
    ##################################################
  }else if (measure == 'standard_pmse'){

    result <- utility.gen2(x, x_link, resamp.method = resamp.method)$standard_pMSE

    ##################################################
    # alpha-precision
    ##################################################
  }else if (measure == 'alpha_precision'){

    result <- compute_metrics(real_data = x, synthetic_data = x_link)[[1]]$Dpa

    ##################################################
    # beta-recall
    ##################################################
  }else if (measure == 'beta_recall'){

    result <- compute_metrics(real_data = x, synthetic_data = x_link)[[1]]$Dcb

    ##################################################
    # authenticity score
    ##################################################
  }else if (measure == 'authen'){

    result <- compute_metrics(real_data = x, synthetic_data = x_link)[[1]]$mean_aut

    ##################################################
    # Cl overlap
    ##################################################
  }else if (measure == 'cl_overlap'){

    if(is.null(cl_target)){
      stop("please check cl_target")
    }

    if(isTRUE(cl_measure)){
      cl_measure <- 'True'
    }else if(isFALSE(cl_measure)){
      cl_measure <- 'FALSE'
    }

    result <- CISynthpop(original_data = x, synthetic_data = x_link, target=cl_target, Measure=cl_measure, m=cl_m)

    ##################################################
    # IDR
    ##################################################
  }else if (measure == 'idr'){

    if (is.null(sensitive_var)){
      stop("please check sensitive_var")
    }

    # if (is.null(sensitive_conti_k)){
    #   sensitive_conti_k <- ifelse(sapply(x[sensitive_var], is.numeric), k, 0)
    # }

    result <- Disclosure(orig = x, syn = x_link, sensitive_var=sensitive_var, sensitive_conti_k=sensitive_conti_k)

    ##################################################
    # CAP
    ##################################################
  }else if (measure == 'cap'){

    if (is.null(sensitive_var)){
      stop("please check sensitive_var")
    }

    result <- CAP(original_data = x, synthetic_data = x_link, attribute=sensitive_var, conti_var=conti_var, K_list=K_list)

    ##################################################
    # Population uniqueness
    ##################################################
  }else if (measure == 'pop_uni'){

    if (!conti_var){
      conti_var <- c()
    }

    result <- pop_uni(original_data = x, synthetic_data = x_link, conti_var=conti_var, K_list=K_list)

    ##################################################
    # Record Linkage
    ##################################################
  }else if (measure == 'record_linkage'){

    doParallel::registerDoParallel(cl <- parallel::makeCluster(parallel::detectCores()-1))

    if (is.null(scale)){
      scale <- rep(1, length(quasi_var))
    }

    if (!n_sampling){
      n_sampling <- nrow(x)/2
    }

    if (is.null(thres)){
      thres <- length(quasi_var)
    }

    fs_result <- c()
    for(i in 1:iteration){
      set.seed(i)

      random_indices <- sample(rownames(x), size = n_sampling, replace = FALSE)

      x_sample <- x[random_indices,]
      x_link_sample <- x_link[random_indices,]

      features <- record_linkage(x = x_sample, x_link = x_link_sample, block = sensitive_var, scale = scale)
      fs_result <- c(fs_result, sum(features$sum >= thres))
    }
    
    parallel::stopCluster(cl)

    result <- mean(fs_result)/n_sampling**2

    ##################################################
    # Density Overfit Membership Inference Attack
    ##################################################
  }else if(measure == 'mia'){

    if (is.null(thres)){
      thres <- 2
    }

    result <- compute_mia(original_data = x, synthetic_data = x_link, threshold = thres)$prop

    ##################################################
    # Distance to Closest Record, Nearest Neighbor Distance Ratio
    ##################################################
  }else if(measure == 'dcr' | measure == 'nndr'){

    if (is.null(col_info)) {
      col_info <- list(cont = which(sapply(x, is.numeric)),
                       ordi = which(sapply(x, is.ordered)),
                       cate = which(!sapply(x, function(x) is.numeric(x)|is.ordered(x))))
    }

    tmp <- DCR_NNDR(original_data = x, synthetic_data = x_link, col_info = col_info)

    if (measure == 'dcr'){
      result <- list(DCR_mean = mean(tmp$DCR, na.rm=TRUE), DCR = tmp$DCR)
    }else if (measure == 'nndr'){
      result <- list(NNDR_mean = mean(tmp$NNDR, na.rm=TRUE), NNDR = tmp$NNDR)
    }

  }else {
    stop("please select your measure from the following: 'w_distance', 'kl_divergence', pmse', 'standard_pmse', 'alpha_precision', 'beta_recall', 'authen', 'cl_overlap', 'idr, 'cap', 'pop_uni', 'record_linkage', 'mia', 'dcr', 'nndr'")
  }

  return(result)

}