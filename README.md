# Causal Inference Using Markov Blanket

## 1. ABSTRACT
Causality, in simple terms, can be described as the agency that connects one process
(the cause) with another (the effect), where the first process is responsible for the
second, and the second is dependent on the first. The causal inference problem
usually consists of some features and a target variable. The goal of the problem is to
identify the causes (features) of the target.

## 2. INTRODUCTION
Over the course of this Independent Project, we worked on two methods that
contribute towards causal inference for any given dataset. Both methods can
generally be categorized as feature-selection methods. The first method uses the Markov
blanket technique (of a target variable) for causal discovery [Brown et al. and Pellet et al.],
whereas the second method uses linear SVM for feature selection [Chang et
al.]. The feature set can be converted into a directed cause-effect graph, using
which the causal inference for the target can be deduced. The Markov blanket
consists of all the features in the cause-effect graph, which are called parents,
children, and spouses of the target variable. The parents are the features that cause
the target, the children are caused by the target, and the spouses
are the features equally responsible for the causation of the children.

## 3. METHODS
Causal inference using Markov blanket
The implementation of this method was majorly based on the following paper:
[Laura E. Brown et al.; A Strategy for Making Predictions Under Manipulation; 2008]
with references to the following papers:
[1] Gavin C. Cawley; Causal and Non-Causal Feature Selection for Ridge
Regression; 2009.
[2] Jianxin Yin et al. Partial orientation and local structural learning of causal
networks for prediction; 2008.
[3] Markus Kalisch et al. Causal inference using graphical models with the R
package pcalg; 2012.
[4] Jean-Philippe Pellet et al.; Using Markov blankets for causal structure learning;
2008.

We were given the following datasets LUCAS, LUCAP, REGED, SIDO, CINA and
MARTI. The datasets LUCAS, LUCAP, and SIDO have binary data. My strategy was as
follows:

### 3.1.1 Pre-processing
The REGED dataset consisted of discrete values that were normalized in the range
of [-1,1] such that their mean was 0 and std variance was 1. LUCAS, LUCAP and
SIDO did not require any pre-processing.

### 3.1.2 Markov blanket generation
The following algorithms such as PC, TPDA, GS and IAMB can be used to generate
Markov blankets. We chose IAMB because it produces better results under the
faithfulness condition.
The correctness of IAMB is under the assumption of the faithfulness condition which
is as follows:
1. The learning database D is an independent and identically distributed sample from
a probability distribution p faithful to a DAG G. 2. The tests of conditional
independence and the measure of conditional dependence are correct.

### ALGORITHM 1: IAMB (Incremental association Markov Blanket )
#### IAMB(dataset D; target T)

1: MB(T) = Phi <br />
2: V = Set of features in D<br />

##### Growing Phase:<br />
Add true positives to MB(T)<br />

3: Repeat Until MB(T) does not change <br />
4: Find Xmax in V-MB(T)-{T} that maximizes MI(Xmax; T | MB(T)) <br />
5: If (Xmax !⊥ T | MB(T)) then <br />

If Xmax is not independent of T given MB(T)

6: MB(T) = MB(T) Union {Xmax} <br />
7: End If <br />

##### Shrinking Phase:
Remove false positives from MB(T)

8: For each X which belongs to MB(T) do <br />
9: If (X ⊥ T | MB(T) - {X}) then <br />

If X is independent of T given MB(T)

10: MB(T) = MB(T) - {X} <br />
11: End if <br />
12: End For <br />
13: Return MB(T) <br />

Here, ⊥ represents independence condition and MI(X; T | MB(T)) indicates mutual
information between X and T given the set MB(T). MB(T) represents Markov blanket
for the target variable. Mutual information is used as a test for conditional
independence. Other tests such as Chi-square test etc. can also be used.

#### 3.1.3 Conditional Independence test
Below is the algorithm to find conditional independence between two variables
conditioned on a set. <br />

### ALGORITHM 2 : Conditional Independence
##### 1. Initialization: <br />
Set S = “empty set”, set X = ”initial set of all D features”. <br />

##### 2. Pre-computation: <br />
For all features Xi ∈ X compute I (C,Xi).

##### 3. Selection of the first feature:<br />
Find feature X ∈ X that maximizes I (C,Xi) ; set X = X \ {X}, S = {X}.

##### 4. Greedy feature selection:<br />
Repeat until the desired number of features is selected.<br />
(a) Computation of entropy:<br />
For all Xs ∈ S compute entropy H(Xs), if it is not already available.<br />

(b) Computation of the MI between features:<br />
For all pairs of features (Xi, Xs) with Xi ∈ X, Xs ∈ S compute I(Xi, Xs), if it is not yet available.<br />

(c) Selection of the next feature:<br />
Find feature X+ ∈ X according to formula :<br />
X+ = arg maxXi ∈ X\S { I(C,Xi) − maxXs ∈ S CU(Xi,Xs) I(C,Xs)}.<br />
Here, CU(Xi,Xs) = I(Xi, Xs) /H(Xs)<br />

In the above algorithm, I(C, Xi) represents mutual information between C and Xi and
H(Xs) represents entropy of variable Xs.<br />

#### 3.1.4 Combined Feature information
Threshold for the Markov blanket algorithm was kept at 0.01, which was shown to be
an ideal numeric value according to Chapter 6 of Ensembles in Machine Learning
Applications , T. Windeatt.<br />

The results of the above two algorithms produced features which were most likely the
causes of the target variable.

#### 3.1.5 Model construction and results
Once the variable list was determined for each dataset, a SVM classification model
was trained using the training dataset and the results of the test dataset was
predicted. The generated variable list was then used to train another SVM classifier
on the training dataset and the target value of the test dataset was generated for the
same variables. The two target values for the two test datasets (one being he subset
of another) were compared to find the number of matches. Based on the comparison,
the accuracy was predicted as follows:

[No. of features selected (Markov blanket)/ Total features] & [Accuracy of results(in %)]

#### LUCAS0 5/11 95.34 <br />
#### LUCAP0 42/143 97.07 <br />
#### REGED0 8/999 91.85 <br />
#### SIDO0 2/4932 98.67 <br />
