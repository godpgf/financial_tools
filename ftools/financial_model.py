#coding=utf-8
#author=godpgf
import numpy as np
from scipy.stats import norm

class FinancialModel(object):

    @classmethod
    def get_CAMP(cls, market_returns,#市场收益
                 risk_free_returns,#无风险收益
                 asset_returns,#当前stock收益
                 confidence = .05
                 ):
        market_premium = np.atleast_2d(market_returns - risk_free_returns).T
        asset_premium = np.atleast_2d(asset_returns - risk_free_returns).T

        constant = np.ones((market_premium.shape[0], 1))
        covariates = np.concatenate((constant, market_premium), axis=1)

        critical_value = norm.ppf(1 - confidence / 2.0)

        # Solve the capital asset pricing model in the least-squares sense. In
        # particular, wel solve the following linear model for parameters theta_0
        # and theta_1:
        #     R_{j,t} - mu_{f,t} = theta_0 + theta_1 * (R_{M,t} - mu_{f,t}) + e_{j,t}
        # Where R_{j,t} is the asset premium of the jth asset, mu_{f,t} is the
        # risk-free rate, R_{M,t} is the market premium, and e_{j,t} represents an
        # error term. Refer to page 435 in the Statistics and Data Analysis for
        # Financial Engineering.
        theta = np.linalg.lstsq(covariates, asset_premium)[0]
        residuals = asset_premium - np.dot(covariates, theta)

        # The rank of the covariates matrix is presumably two, and it is for that
        # reason that we subtract two in the denominator.
        s_squared = np.sum(residuals * residuals) / (market_premium.shape[0] - 2)

        standard_errors = np.sqrt(s_squared * np.linalg.inv(np.dot(covariates.T, covariates)))
        alpha_value = theta[0]
        alpha_confidence_interval = theta[0] + standard_errors[0, 0] * critical_value * np.array([-1, 1])

        beta_value = theta[1]
        beta_confidence_interval = theta[1] + standard_errors[1, 1] * critical_value * np.array([-1, 1])
        return alpha_value, alpha_confidence_interval, beta_value, beta_confidence_interval

