# MLcF

## 📜 Description

_Machine Learning Lifecycle Framework_ (**MLcF**) is an experimental toolbox implemented in Python that provides instruments 
to create a controlled environment to play around with various data errors and stages in the ML lifecycle, 
and measure their impact on model fairness and stability. To perform a detailed audit of model fairness and stability,
we develop a separate software library called [Virny](https://github.com/DataResponsibly/Virny), which is also one essential component 
for computing model performance metrics in MLcF.

The framework's design was guided by three foundational decisions:
1) easy extensibility and modularity that aligns with our ML lifecycle view and empirical approach; 
2) flexible functionality that creates a playground with controlled error injection; 
3) convenience for conducting multiple experiments with various classification datasets.


## 💡 Features

With our toolbox, users have the ability to perform the following actions:
* inject a controlled level of various data errors into the ML lifecycle (e.g., outliers, mislabels, nulls, incorrect proportions of protected groups, etc.);
* conduct experiments using provided experiment interfaces;
* measure multiple fairness and stability metrics computed by the Virny library;
* save results in preferred storage (e.g., local disk or even own database) and visualize metrics using custom plots generated by the framework.

Therefore, this practical toolset can be used by both ML practitioners and researchers. The former can apply it
to stress-test model fairness and stability under various controlled data errors. Whereas the latter can utilize it to test novel 
fairness/robustness-enhancing interventions under various error injectors and to identify relationships between 
these interventions and specific data errors they can  mitigate.


## 🤗 Affiliations

![NYU-UCU-Logos](https://user-images.githubusercontent.com/42843889/216840888-071bf184-f0e3-4a3e-94dc-c0d1c7784143.png)


## 📝 License

**MLcF** is free and open-source software licensed under the [3-clause BSD license](https://github.com/DataResponsibly/MLcF/blob/main/LICENSE).
