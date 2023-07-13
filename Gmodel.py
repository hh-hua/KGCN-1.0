# class mymodel():
#     def __init__(self, h=None, lam=1,maxiter=500, tol=1e-6):
#         self.beta = None
#         self.h = h
#         self.dataset = None
#         self.maxiter = maxiter
#         self.tol = tol
#         self.funvalue = None
#         self.coef = None
#         self.dataset = None
#         self.lam=lam
#         self.iteration=maxiter
#
#
#     def fit(self, X_train, y_train):
# 		#用于训练模型参数，例如self.coef
#         self.coef, self.funvalue = myfun(X_train, y_train)
#
#     def predict(self, X_new):
# 		#用于根据X预测y，返回y的预测值数组
# 		#XXXXXXX
#         return y_pre
#
#
#     def get_params(self, deep=True):
#         """Get parameters for this estimator.
#
#         Parameters
#         ----------
#         deep : boolean, optional
#             If True, will return the parameters for this estimator and
#             contained subobjects that are estimators.
#
#         Returns
#         -------
#         params : mapping of string to any
#             Parameter names mapped to their values.
#         """
#         out = dict()
#         for key in ['h','lam','maxiter','tol']:#这里是所用超参数的list
#             value = getattr(self, key, None)
#             if deep and hasattr(value, 'get_params'):
#                 deep_items = value.get_params().items()
#                 out.update((key + '__' + k, val) for k, val in deep_items)
#             out[key] = value
#         return out
#
#     def set_params(self, **params):
#         """Set the parameters of this estimator.
#
#         The method works on simple estimators as well as on nested objects
#         (such as pipelines). The latter have parameters of the form
#         ``<component>__<parameter>`` so that it's possible to update each
#         component of a nested object.
#
#         Returns
#         -------
#         self
#         """
#         if not params:
#             # Simple optimization to gain speed (inspect is slow)
#             return self
#         valid_params = self.get_params(deep=True)
#
#
#         for key, value in params.items():
#             if key not in valid_params:
#                 raise ValueError('Invalid parameter %s for estimator %s. '
#                                  'Check the list of available parameters '
#                                  'with `estimator.get_params().keys()`.' %
#                                  (key, self))
#             setattr(self, key, value)
#             valid_params[key] = value
#
#         return self
#     def score(self, X, y, sample_weight=None):
#     	#如果这里不设置score函数，可以在GridSearchCV()的scoring参数中指定
#         """Returns the mean accuracy on the given test data and labels.
#
#         In multi-label classification, this is the subset accuracy
#         which is a harsh metric since you require for each sample that
#         each label set be correctly predicted.
#
#         Parameters
#         ----------
#         X : array-like, shape = (n_samples, n_features)
#             Test samples.
#
#         y : array-like, shape = (n_samples) or (n_samples, n_outputs)
#             True labels for X.
#
#         sample_weight : array-like, shape = [n_samples], optional
#             Sample weights.
#
#         Returns
#         -------
#         score : float
#             Mean accuracy of self.predict(X) wrt. y.
#
#         """
#         return myloss_fun(y, self.predict(X), sample_weight=sample_weight)
