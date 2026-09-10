"""Native signed predictive distributions trained only on past available labels."""
import importlib.metadata
import numpy as np
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from sklearn.tree import DecisionTreeRegressor
from research.eps_model_lab_v1.common import EPSAdapter,features

SPECS={
 'ngboost_normal':{'distribution':'Normal','n_estimators':500,'learning_rate':.01,'natural_gradient':True,'tree_depth':3,'min_samples_leaf':5,'early_stopping_rounds':None},
 'ngboost_laplace':{'distribution':'Laplace','n_estimators':500,'learning_rate':.01,'natural_gradient':True,'tree_depth':3,'min_samples_leaf':5,'early_stopping_rounds':None},
 'bayesian_ridge_distribution':{'max_iter':500,'tol':.001,'priors':'Official defaults, train-only evidence optimization'},
 'gaussian_process_matern_distribution':{'kernel':'Matern nu1.5 lengthscale=sqrt(number_of_train_transformed_features) + WhiteKernel0.1',
    'optimizer':None,'alpha':1e-6,'normalize_y':True},
}


class ProbabilisticAdapter(EPSAdapter):
    def __init__(self,name,smoke=False):
        self.name=name;self.smoke=smoke
        package='ngboost' if name.startswith('ngboost') else 'scikit-learn'
        super().__init__(name,{'family':'NGBOOST_DISTRIBUTIONAL' if package=='ngboost' else 'BAYESIAN_PREDICTIVE_DISTRIBUTION',
          'version':importlib.metadata.version(package),'pit_valid':True,'zero_shot':False,'finetuned':False,
          'source':'https://github.com/stanfordmlgroup/ngboost' if package=='ngboost' else 'https://scikit-learn.org/stable/',
          'license':'Apache-2.0' if package=='ngboost' else 'BSD-3-Clause','spec':SPECS[name],
          'preprocessing':'Common scaled features; fold-train median imputer and StandardScaler. Target divided by causal origin scale.',
          'uncertainty':'Native fitted conditional distribution, not a posthoc invented interval'})
    def prepare_data(self,frame):return features(frame,scaled=True).to_numpy(dtype=np.float64)
    def fit(self,frame,target):
        self.target=target;self.columns=list(features(frame).columns)
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=StandardScaler();x=self.scaler.fit_transform(self.imputer.fit_transform(self.prepare_data(frame)))
        y=(frame['y_'+target]/frame.scale).to_numpy()
        if self.name.startswith('ngboost'):
            from ngboost import NGBRegressor
            from ngboost.distns import Normal,Laplace
            self.model=NGBRegressor(Dist=Normal if self.name=='ngboost_normal' else Laplace,
              Base=DecisionTreeRegressor(max_depth=3,min_samples_leaf=5,random_state=1729),
              n_estimators=5 if self.smoke else 500,learning_rate=.01,natural_gradient=True,
              random_state=1729,verbose=False,early_stopping_rounds=None)
        elif self.name=='bayesian_ridge_distribution':self.model=BayesianRidge(max_iter=500,tol=.001)
        else:
            kernel=Matern(length_scale=np.sqrt(x.shape[1]),length_scale_bounds='fixed',nu=1.5)+WhiteKernel(noise_level=.1,noise_level_bounds='fixed')
            self.model=GaussianProcessRegressor(kernel=kernel,optimizer=None,alpha=1e-6,normalize_y=True,random_state=1729)
        self.model.fit(x,y);self.fitted=True;self.metadata['fitted_target']=target;return self
    def distribution(self,frame):
        if list(features(frame).columns)!=self.columns:raise RuntimeError('Feature order drift')
        x=self.scaler.transform(self.imputer.transform(self.prepare_data(frame)));scale=frame.scale.to_numpy()
        if self.name.startswith('ngboost'):
            d=self.model.pred_dist(x);p=d.mean();q=np.column_stack([d.ppf(t) for t in [.1,.5,.9]])
        else:
            p,std=self.model.predict(x,return_std=True)
            if (std<0).any():raise RuntimeError('Negative native predictive standard deviation')
            q=p[:,None]+std[:,None]*norm.ppf([.1,.5,.9])[None,:]
        return p*scale,q*scale[:,None]
    def predict(self,frame,target):
        if target!=self.target:raise RuntimeError('Target mismatch')
        return self.distribution(frame)[0]
