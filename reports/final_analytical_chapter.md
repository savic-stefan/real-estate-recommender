# Final Analytical Chapter

## Core Result
- Best price model: Random Forest Regressor
- Best recommendation method: content_weighted
- Final decision-support shortlist size: 15
- Top shortlisted listing: 11424

## Stage Outcomes
| stage | headline | key_result |
| --- | --- | --- |
| Stage 4 - Preprocessing | Data quality pipeline | 15653414 clean rows; 300000 modeling rows |
| Stage 6 - Price Evaluation | Best market valuation model | Random Forest Regressor with RMSE 3863765.0396 and R2 0.816416 |
| Stage 7 - Feature Engineering | Engineered feature impact | 19 baseline features vs 26 enhanced features; RMSE delta 64523.3477 |
| Stage 12 - Recommendation Evaluation | Best recommendation method | content_weighted with DCG 4.091525 |
| Stage 11 - Hybrid Recommendation | Hybrid integration result | 3375 gated candidates; collaborative signal remained non-zero but small |
| Stage 13 - Decision Support | Integrated shortlist | 4608 feasible candidates; 15 final shortlist objects |
| Stage 14 - Final Demo | Defense-ready packaging | Top listing 11424 collected into final demo payload |

## Key Findings
| finding_id | theme | statement | evidence |
| --- | --- | --- | --- |
| F1 | Price modeling | Random Forest Regressor was the best market valuation model on holdout data. | test_rmse=3863765.0396; test_r2=0.816416 |
| F2 | Price modeling | Nonlinear ensembles substantially outperformed the linear baseline. | RMSE improvement over Linear Regression=46.7940% |
| F3 | Feature engineering | Engineered domain features were informative but did not improve direct price regression. | enhanced_delta_rmse=64523.3477; enhanced_delta_r2=-0.006454 |
| F4 | Recommendation | The weighted content-based recommender achieved the best offline utility. | method=content_weighted; dcg_utility=4.091525 |
| F5 | Recommendation | Hybrid and knowledge-based methods perfectly satisfied hard constraints, but did not surpass content_weighted on utility. | hybrid_constraint_rate=1.0000; knowledge_constraint_rate=1.0000 |
| F6 | Hybrid integration | Collaborative signal remained the weakest component inside the hybrid recommendation stage. | hybrid_collaborative_component_mean=0.013665 |
| F7 | Decision support | The integrated shortlist contained market-underpriced objects relative to the learned valuation model. | top_listing_discount=0.417539; top_listing_score=0.756789 |
| F8 | Decision support | The final DSS pipeline reduced a large candidate pool to a compact shortlist suitable for human review. | candidate_pool=4608; shortlist=15 |

## System Limitations
| limitation_id | area | issue | evidence |
| --- | --- | --- | --- |
| L1 | Data quality | A large share of shortlist objects still has missing categorical semantics. | unknown_metro=13/15; unknown_renovation=13/15 |
| L2 | Collaborative filtering | Collaborative recommendation quality is constrained by synthetic and sparse feedback. | item_based_dcg=2.955101; user_based_dcg=2.929362 |
| L3 | Knowledge-based rules | Exact soft preference rules were too restrictive to apply safely in the candidate pool. | optional_rules_applied=0 |
| L4 | Feature engineering | Rule-based engineered features did not improve direct market price prediction in the current setup. | delta_rmse_vs_baseline=64523.3477 |

## Shortlist Preview
| rank | listing_id | Price | predicted_market_price | relative_discount_to_market | decision_support_score | explanation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11424 | 5620000.0000 | 9648711.3816 | 0.4175 | 0.7568 | below_estimated_market_price; high_investment_potential; high_liquidity |
| 2 | 9182 | 11215458.0000 | 15295308.9106 | 0.2667 | 0.7559 | below_estimated_market_price; strong_preference_match |
| 3 | 3677 | 9208220.0000 | 15071585.7286 | 0.3890 | 0.7496 | below_estimated_market_price; high_liquidity |
| 4 | 19196 | 3400000.0000 | 5603943.2414 | 0.3933 | 0.7414 | below_estimated_market_price; high_investment_potential; high_liquidity |
| 5 | 5135 | 7400000.0000 | 11097201.1062 | 0.3332 | 0.7356 | below_estimated_market_price; strong_preference_match; preferred_renovation_match |

## Analytical Conclusion
The project demonstrates that nonlinear market valuation and weighted content-based recommendation provide the strongest empirical foundation for the housing decision-support system. The integrated DSS shortlist is practically useful because it combines preference fit with valuation-based discount signals. At the same time, the experiments show that data sparsity, synthetic collaborative feedback, and missing categorical values still constrain the upper bound of recommendation quality.
