Collaborative Federated Multi-Modal Learning for Cross-Institutional Emergency Triage
 
Name Surname1🖂, Name Surname2, Name Surname3
 
1  …….. Department, ……… University, City, Country
2  ……… Department, ……… University, City, Country
 
Abstract
Emergency departments require accurate, data-enabled triage models to maximize patient throughput and prioritization of care. Training such models on diverse, multi-site data is limited. Training patient encounters across institutions together is typically blocked by strict privacy statutes as well as cumbersome governance constraints, which hinders the creation of generalizable and strong AI systems. To remedy this, we propose a case study of federated learning that shows collaborative model training is possible under data-local, privacy-aware conditions.  
 
In this research, a multi-path, attention-compiled neural network was trained over five simulated hospitals. The model incorporated the large-scale Kaggle hospital triage dataset, with 558,029 encounters, with all the data remaining safely local. The FL approach ensures that no raw data are shared; rather, model parameters are aggregated to construct collective intelligence. The sites used the same feature engineering pipeline, building inputs from augmented vital signs, universal laboratory values, as well as clinically pertinent interaction features. Each site also optimized a focal–penalty objective function.  The results are encouraging. The centralized baseline model trained on pooled data obtained 88.86% accuracy. Our federated variant, under privacy-aware conditions, obtained a competitive 86.6% accuracy in five rounds of collaborative learning. Per-class accuracy was maintained high, with precision and recall rates of 97.3% and 78.4% for low-risk (Green), 81.8% and 93.1% for moderate-risk (Yellow), and 90.5% and 82.2% for high-risk (Red) instances. Post-training calibration, such as temperature scaling as well as validation-based threshold sweeping, enhanced predictive accuracy, with the confusion matrix obtained being similar to the centralized reference one. Validation accuracy progressed steadily over the rounds of training with successful convergence of the collaborative protocol.  
 
These findings demonstrate that near-centralized triage performance is possible while maintaining strict data-local constraints. This research establishes a strong foundation for future cross-institutional collaboration toward the right cause in emergency care, enabling the possibility of being more correct and fair with AI in healthcare.
 
Keywords: federated learning, medical triage
 
References
[1] H. B. McMahan, E. Moore, D. Ramage, et al., “Communication-efficient learning of deep networks from decentralized data,” Proc. AISTATS, 2017.
 
[2] N. Rieke et al., “The future of digital health with federated learning,” NPJ Digital Medicine, 2020.
 
[3] M. J. Sheller et al., “Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data,” Scientific Reports, 2020.
 
[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” Proc. ICML, 2017.
 
[5] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” Proc. ICCV, 2017.
 
[6] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Proc. NeurIPS, 2017.
 
[7] M. Hardt, E. Price, and N. Srebro, “Equality of opportunity in supervised learning,” Proc. NeurIPS, 2016.
 
[8] I. Dayan, H. Roth, A. Zhong, et al., “Federated learning for predicting clinical outcomes in patients with COVID-19,” Nature Medicine, vol. 27, pp. 1735–1743, 2021.
 
[9] Y. Raita, T. Goto, M. K. Faridi, D. F. M. Brown, C. A. Camargo Jr., and K. Hasegawa, “Emergency department triage prediction of clinical outcomes using machine learning models,” Critical Care, vol. 23, p. 64, 2019.
 
[10] Y. Liu, J. Gao, J. Liu, et al., “Development and validation of a practical machine-learning triage algorithm for the detection of patients in need of critical care in the emergency department,” Scientific Reports, vol. 11, p. 24044, 2021.
 
[11] Y.-H. Chang, Y.-C. Lin, F.-W. Huang, et al., “Using machine learning and natural language processing in triage for prediction of clinical disposition in the emergency department,” BMC Emergency Medicine, vol. 24, p. 237, 2024.
 
[12] D. Gligorijevic, J. Stojanovic, W. Satz, et al., “Deep attention model for triage of emergency department patients,” arXiv:1804.03240, 2018.
 
[13] A. J. Vickers and E. B. Elkin, “Decision curve analysis: a novel method for evaluating prediction models,” Medical Decision Making, vol. 26, no. 6, pp. 565–574, 2006.
 
[14] I. Araf, S. I. Ayon, and A. Rahman, “Cost-sensitive learning for imbalanced medical data: a review,” Artificial Intelligence Review, 2024.
 
[15] B. Pes, A. P. Ruiu, G. Cossu, and G. Angioni, “Cost-sensitive learning strategies for high-dimensional and imbalanced medical data,” BMC Medical Informatics and Decision Making, vol. 21, 2021.
 
[16] X. Tong, “A plug-in approach to Neyman–Pearson classification,” Journal of Machine Learning Research, vol. 14, pp. 3011–3040, 2013.
 
[17] L. Wang, Y. Chen, A. Jiang, et al., “Hierarchical Neyman–Pearson classification for prioritizing low-false-negative decisions,” Journal of the American Statistical Association, 2024.
 
[18] Y. Ovadia, E. Fertig, J. Ren, et al., “Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift,” Proc. NeurIPS, 2019.
 
[19] H. Peng, W. Wang, F. Chen, et al., “FedCal: Achieving local and global calibration in federated learning via aggregated parameterized scaler,” Proc. ICML, PMLR 235, 2024.
 
[20] Y. Yu, J. Ma, H. Zhao, and N. Vasconcelos, “Robust calibration with multi-domain temperature scaling,” Proc. NeurIPS, 2022.
 
[21] Z. L. Teo, L. Jin, S. Li, D. Miao, and D. S. W. Ting, “Federated machine learning in healthcare: A systematic review on clinical applications and technical architecture,” Cell Reports Medicine, vol. 5, no. 2, p. 101419, 2024.
 
⸻
 
 
🖂 Corresponding Author Email: name.surname@email.edu / name.surname@org / name.surname@email
https://iww.itu.edu.tr/