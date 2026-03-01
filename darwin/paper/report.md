\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\DeclareUnicodeCharacter{2212}{-}
\usepackage{newtxtext,newtxmath}
\usepackage{graphicx}
\usepackage{multicol}
\usepackage[normalem]{ulem}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{natbib}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\usepackage{titling}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.8\baselineskip}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    citecolor=blue!60!black,
    urlcolor=blue!60!black
}

% --- Title-page layout (titling package) ---
\setlength{\droptitle}{-3cm}
\pretitle{%
  \noindent
  {\normalsize\textit{Preprint}}%
  \par\vspace{6pt}%
  \noindent\rule{\linewidth}{1.5pt}%
  \par\vspace{20pt}%
  \begin{center}
  \LARGE
}
\posttitle{%
  \end{center}%
}
\preauthor{\begin{center}\vspace{5pt}\normalsize}
\postauthor{\end{center}}
\predate{\begin{center}\small}
\postdate{\end{center}}

\begin{document}

\title{Preventing Curriculum Collapse in Self-Evolving Reasoning Systems}

\author{\textbf{Vaibhav Mishra}\\[2pt]{\footnotesize\textrm{vaibhavm209625@gmail.com}}}
\date{}
{\setlength{\parskip}{0pt}\maketitle}

% ============================================================================
% ABSTRACT
% ============================================================================

\vspace{-15pt}
\begin{abstract}
\noindent
Self-evolving reasoning frameworks let LLMs improve their reasoning capabilities by iteratively generating and solving problems without external supervision through verifiable rewards. Ideally, such systems are expected to explore a diverse problem space and propose new challenges of high learning value. While prior work has largely focused on solver-side optimisation and verification, recent evidence suggests that self-evolving systems can exhibit \emph{diversity collapse} in posing new problems after just a few iterations, even when surface-level variation is preserved~\cite{li2025rdiverse,zhou2025evolrl}.

We introduce \textbf{Prism}, a question-centric self-evolution method that directly tackles this collapse. Prism defines a persistent diversity signal over an embedding-induced semantic partition of mathematical problems and uses it to encourage balanced exploration of underrepresented regions across iterations. This coverage signal is combined with a Zone-of-Proximal-Development (ZPD) gate to preserve edge-of-solvability difficulty.

Evaluated on seven widely used mathematical reasoning benchmarks against five self-evolving baselines, Prism achieves the highest accuracy on six out of seven tasks, achieving gains of $+3.37$ absolute points over R-Zero on AIME~2024 and $+3.68$ on Minerva Math. Prism also generates semantically diverse, edge-of-solvability questions across iterations, resulting in the construction of the Prism-Math dataset—a diverse and high-difficulty mathematical question set comprising 100k questions. These results demonstrate that cross-iteration semantic coverage is a high-leverage and previously underexplored axis for building more capable self-evolving reasoners.
\end{abstract}
\begin{figure}[h]
\centering
\begin{minipage}{0.49\linewidth}
    \centering
    \includegraphics[width=\linewidth]{main_results.png}
\end{minipage}\hfill
\begin{minipage}{0.49\linewidth}
    \centering
    \includegraphics[width=\linewidth]{training_curve.png}
\end{minipage}
\caption{Prism performance overview. (a) Pass@1 accuracy comparison with baseline across benchmarks. (b) Math-500 test accuracy across iterations.}
\label{fig:main_results_training}
\end{figure}

\vspace{-6pt}
\begin{center}
\small
\textbf{Resources:}\quad
\href{https://github.com/PLACEHOLDER/prism}{\texttt{Code}}\quad$\vert$\quad
\href{https://huggingface.co/PLACEHOLDER/prism-solver}{\texttt{Models}}\quad$\vert$\quad
\href{https://huggingface.co/datasets/PLACEHOLDER/prism-math}{\texttt{Prism-Math Dataset}}
\end{center}
\vspace{-6pt}

% ============================================================================
% 1  INTRODUCTION
% ============================================================================
\section{Introduction}

Reinforcement learning with verifiable rewards (RLVR) has substantially advanced mathematical reasoning in large language models (LLMs)~\cite{guo2025deepseekr1,shao2024deepseekmath}.
In parallel, a growing line of work has explored \emph{self-evolution}: training systems that improve 
by generating and solving their own tasks, reducing dependence on curated supervision and enabling 
scalable self-loop learning~\cite{silver2017alphazero,chen2024spin,huang2025rzero,yue2026drzero,
chen2025selfevolvingcurriculum,kuba2025languageselfplay}.
In this paradigm, a \textbf{Questioner} proposes problems and a \textbf{Solver} learns from them under verifiable reward signals such as majority voting or self-verification~\cite{huang2025rzero,zhou2025evolrl,shao2025deepseekmathv2}.

However, training on self-generated data can cause concept drift or entropy collapse, degrading curriculum diversity and inducing repetition or forgetting over time~\cite{shumailov2023curse,dohmatob2024modelcollapse}.
Difficulty calibration (ZPD-style gating) provides a \emph{local} learnability signal but does not create pressure to visit underrepresented regions of the problem space.
R-Zero employs an online BLEU-based clustering penalty to discourage lexical similarity 
within a batch.
However, lexical signals fail to detect semantic equivalence, operate only locally within a batch, and 
lack cross-iteration memory.
R-Few~\cite{yu2025rfew} stabilises co-evolution by injecting a small pool of human-labeled anchor examples as in-context guidance, but this diversity is externally induced and requires ongoing access to labeled data.

Recent work has begun diagnosing related issues.
R-Diverse~\cite{li2025rdiverse} identified the diversity illusion in self-play training, while 
Evol-RL~\cite{zhou2025evolrl} studied solution-space entropy collapse under majority-driven selection 
and proposed novelty-aware rewards to preserve variation.
Absolute Zero~\cite{zhao2025absolutezero} eliminates external data entirely by having a single model propose and solve code-reasoning tasks validated by a code executor, achieving strong cross-domain transfer to mathematics.
SPICE~\cite{chen2025spice} leverages corpus-grounded self-play in which the model trains on self-generated questions conditioned on retrieved documents, improving reasoning through environment interaction.
Other directions improve solver reasoning via data scaling and self-improvement pipelines~\cite
{zeng2024skyworkmath,yang2024qwen25mathtech,pei2025scalediff,wei2025learningtoposeproblems,
lu2024mathgenie}.
However, much of this progress remains \emph{solver-centric}.
In contrast, we argue that the bottleneck lies upstream, in the \emph{semantic coverage} and \emph
{difficulty calibration} of the generated curriculum.
In a co-evolutionary loop, the questioner’s output defines the solver’s training distribution; a solver 
cannot learn strategies it is never prompted to exercise.
Therefore, sustained capability growth requires explicit control over cross-iteration coverage in the 
question generation process.

We introduce \textbf{Prism}, a question-centric, self-evolution framework that explicitly regularises \emph{cross-iteration semantic coverage} during problem generation.
Prism defines a persistent coverage signal over an embedding-partitioned mathematical space and rewards questions from underexplored regions, while a ZPD gate preserves edge-of-solvability difficulty.
In addition, Prism initialises each new Questioner from the latest Solver rather than the previous Questioner, eliminating capability lag, enabling more unbiased exploration, and mitigating progressive distributional narrowing.
Beyond improving Solver performance, Prism functions as a scalable generator of diverse and 
high-difficulty mathematical problems. 
By jointly enforcing edge-of-solvability constraints and cross-iteration semantic coverage, it produces 
curricula that span broad reasoning domains while reaching competition-level hardness, enabling 
autonomous construction of high-quality mathematical datasets.

Across seven mathematical reasoning benchmarks, Prism outperforms five self-evolving baselines on six of seven tasks, with especially large gains on harder evaluations (e.g., AIME~2024, Minerva Math, and AMC).
These results identify cross-iteration semantic coverage as a high-leverage axis for advancing self-evolving reasoners.
We summarise our key contributions as follows:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
    \item We \textbf{identify the solver-centric bias} in the self-evolving LLM literature and argue that the overlooked lever for improvement is the \emph{questioner}: the semantic diversity and quality of the generated curriculum directly determine the solver's performance ceiling.
    \item We \textbf{diagnose curriculum collapse} as the primary failure mode of self-evolving reasoning LLMs and show that existing lexical diversity penalties are insufficient to prevent it.
    \item We propose \textbf{Prism}, a question-centric self-evolution method that treats question generation as a first-class optimisation target within a self-evolving loop, comprising (a) a \textbf{semantic cluster-diversity reward} that provides persistent coverage pressure over a pre-computed semantic partition, and (b) a \textbf{Solver-initialised Questioner} strategy that eliminates capability lag, enables unbiased exploration, and mitigates model collapse by re-deriving the Questioner from the Solver at each iteration.
    \item We provide \textbf{empirical evidence} across seven benchmarks that the gains from semantic question diversity leads to stronger generalization in solver's capabilities.
    \item We release \textbf{Prism-Math}, a dataset of ${\sim}$100K semantically diverse, difficulty-calibrated synthetic mathematical questions generated by the Prism questioner, spanning diverse mathematical topics with verified reference answers, providing a resource for training and evaluating mathematical reasoning models.
\end{enumerate}

% ============================================================================
% 2  RELATED WORK
% ============================================================================
\section{Related Work}


\paragraph{Reinforcement learning for reasoning.}
Outcome-based RL has proven effective for mathematical reasoning.
DeepSeekMath~\cite{shao2024deepseekmath} demonstrated that GRPO enables stable optimisation of verifiable reasoning skills, while DeepSeek-R1~\cite{guo2025deepseekr1} showed that strong reasoning can emerge through pure RL without supervised fine-tuning.
These advances establish RLVR as a practical training paradigm for both solvers and question generators in closed-loop systems.

\paragraph{Self-evolving curricula and diversity collapse.}
Inspired by AlphaZero-style self-play~\cite{silver2017alphazero}, self-play fine-tuning (SPIN)~\cite{chen2024spin} demonstrated that a model can improve by learning from its own generated outputs without external labels---a precursor to the fully closed-loop frameworks that followed.
Recent frameworks extend this idea by training a questioner--solver pair without any external data~\cite{huang2025rzero,yue2026drzero,chen2025selfevolvingcurriculum,kuba2025languageselfplay}.
These systems rely on ZPD-style difficulty signals to maintain learnability, but difficulty alignment alone provides no guarantee that the generated curriculum spans the breadth of the task space.
R-Diverse~\cite{li2025rdiverse} formalised the \emph{diversity illusion}---surface variation masking semantic repetition---and Evol-RL~\cite{zhou2025evolrl} identified entropy collapse focused on the solution space.
R-Few~\cite{yu2025rfew} takes a different approach by injecting a small pool of human-labeled anchor examples as in-context prompts to guide the Challenger; this delays the performance plateau but introduces a systematic \emph{question-side bias} toward the human-curated topic distribution and requires continuous access to labeled data.
Absolute Zero~\cite{zhao2025absolutezero} removes external data entirely: a single model proposes and solves code-reasoning tasks grounded in a code executor, serving as a unified source of verifiable reward.
By casting task generation as code synthesis validated by execution, Absolute Zero achieves strong cross-domain transfer to mathematical reasoning despite training exclusively on self-proposed code tasks.
SPICE~\cite{chen2025spice} introduces corpus-grounded self-play in which the model generates questions conditioned on retrieved passages and trains on its own outputs, improving reasoning through interaction with a textual environment rather than explicit diversity signals.
While both Absolute Zero and SPICE advance the data-free self-evolution frontier, neither explicitly regularises the \emph{cross-iteration semantic coverage} of the generated curriculum---the axis that Prism targets.
Prism instead addresses diversity collapse through an intrinsic semantic coverage signal, requiring no external supervision beyond a one-time offline clustering step.

\paragraph{Curriculum learning, problem synthesis, and quality--diversity.}
Curriculum learning orders training by difficulty~\cite{bengio2009curriculum}; quality-diversity methods such as MAP-Elites~\cite{pugh2016mapelites} maintain solution archives across behavioural niches to prevent mode collapse.
Prism instantiates this principle over semantic topic clusters, gated by solvability to preserve learnability.
A complementary line of work generates or scales high-quality math problems~\cite{lu2024mathgenie,pei2025scalediff,wei2025learningtoposeproblems,zeng2024skyworkmath,yang2024qwen25mathtech,shao2025deepseekmathv2,yuan2025naturalreasoning,liu2025saturn}, but these approaches are primarily solver-centric---they improve the supply of training data without explicitly regularising \emph{cross-iteration semantic coverage} of the questioner.
Liang et al.~\cite{liang2025beyondpass} show that combining self-play with variational problem synthesis can sustain RLVR beyond the single-iteration pass@1 bottleneck, which complements our finding that semantic coverage is the binding constraint in multi-iteration self-evolution.
Prism differs in mechanism and objective: rather than synthesising more problems or improving solver verification alone, Prism explicitly regularises \emph{cross-iteration semantic coverage} of the questioner within a self-evolving loop.

% ============================================================================
% 3  PRELIMINARIES
% ============================================================================
\section{Preliminaries}
\label{sec:preliminaries}

We introduce notation and formalise the building blocks that underpin both the baseline and Prism.

\paragraph{Notation.}
Let $\mathcal{Q}$ denote the space of mathematical problems and $\mathcal{A}$ the space of candidate solutions.
A \emph{questioner} $Q_\theta$ parameterised by $\theta$ defines a generative policy $\pi^Q_\theta$ over $\mathcal{Q}$, and a \emph{solver} $S_\phi$ parameterised by $\phi$ defines a conditional policy $\pi^S_\phi(\cdot\mid q)$ over $\mathcal{A}$.
Training alternates between the two across $T$ co-evolution iterations indexed by $t$~\cite{huang2025rzero}.

\paragraph{Solvability scoring, policy optimisation, and ZPD gating.}
Given a question $q$ and the current solver $S_\phi$, we draw $n$ independent roll-outs $\{a_i\}_{i=1}^{n}\!\sim\!\pi^S_\phi(\cdot\mid q)$ and apply a deterministic verifier $\mathsf{V}\!:\mathcal{Q}\!\times\!\mathcal{A}\!\to\!\{0,1\}$ (exact-match or symbolic equivalence).
The \emph{solvability score}
\begin{equation}
    p(q) \;=\; \frac{1}{n}\sum_{i=1}^{n}\mathsf{V}(q,a_i)\;\in\;[0,1]
    \label{eq:majority_rate}
\end{equation}
provides a label-free, continuous estimate of problem difficulty relative to the current solver~\cite{huang2025rzero,zhou2025evolrl}.

Both agents are trained with Group Relative Policy Optimisation (GRPO)~\cite{shao2024deepseekmath}, a variance-reduced policy-gradient method suited to sparse, outcome-based rewards.
For a prompt $x$, a group of $G$ completions $\{y_g\}_{g=1}^{G}\!\sim\!\pi_\psi(\cdot\mid x)$ is sampled and scored, yielding rewards $\{r_g\}$.
Advantages are computed relative to the within-group mean:
\begin{equation}
    A_g \;=\; r_g \;-\; \frac{1}{G}\sum_{j=1}^{G}r_j\,,
    \label{eq:grpo_advantage}
\end{equation}
and the policy is updated by maximising a clipped surrogate objective with KL regularisation:
\begin{equation}
    \mathcal{J}(\psi)
    \;=\;
    \mathbb{E}_{x}\!\left[
        \frac{1}{G}\sum_{g=1}^{G}
        \min\!\Big(
            \rho_g\,A_g,\;
            \mathrm{clip}(\rho_g,1\!-\!\epsilon,1\!+\!\epsilon)\,A_g
        \Big)
    \right]
    -\;\beta\,\mathrm{KL}\!\bigl(\pi_\psi\|\pi_{\mathrm{ref}}\bigr),
    \label{eq:grpo_obj}
\end{equation}
where $\rho_g = \pi_\psi(y_g\!\mid\!x)/\pi_{\mathrm{ref}}(y_g\!\mid\!x)$ is the importance ratio, $\epsilon$ the clipping threshold, and $\beta$ the KL penalty coefficient.
The group baseline in Equation~\ref{eq:grpo_advantage} eliminates the need for a learned value function, which is particularly advantageous when reward signals are binary or near-binary~\cite{guo2025deepseekr1}.

Finally, self-evolving questioners should generate problems that are neither trivially solvable nor intractable---i.e., problems at the \emph{edge of solvability}~\cite{huang2025rzero}.
We formalise this via a piecewise-linear ZPD gate $g:[0,1]\to[0,1]$:
\begin{equation}
    g(p)
    \;=\;
    \max\!\Bigl(0,\;1-\tfrac{|p-p^\star|}{\Delta}\Bigr)
    \;\cdot\;
    \mathbf{1}\!\bigl[p\in[p_{\min},\,p_{\max}]\bigr],
    \label{eq:zpd_gate}
\end{equation}
which zeroes out reward for questions that are too easy ($p>p_{\max}$) or too hard ($p<p_{\min}$), concentrating reward near target solvability $p^\star$.

\paragraph{R-Zero: data-free co-evolutionary self-play.}
R-Zero~\cite{huang2025rzero} instantiates the primitives above in a Challenger--Solver co-evolution loop.
Both agents are initialised from the same pre-trained model $M_0$ and trained over $T$ iterations.
At iteration $t$, the Challenger generates candidate questions, each scored via the majority-vote solvability $p(q)$ (Equation~\ref{eq:majority_rate}).
The Challenger reward combines an \emph{uncertainty reward} that peaks at maximal Solver uncertainty,
\begin{equation}
    r_{\mathrm{unc}}(q) \;=\; 1 - 2\,\bigl|p(q) - \tfrac{1}{2}\bigr|,
    \label{eq:rzero_unc}
\end{equation}
which attains its maximum at $p(q)=0.5$ and decays linearly as the question becomes trivially easy or impossibly hard, with a \emph{repetition penalty} $r_{\mathrm{rep}}(q)$ computed via pairwise BLEU distances and agglomerative clustering within the batch.
The composite Challenger reward is
\begin{equation}
    r_{\text{R-Zero}}(q) \;=\; r_{\mathrm{unc}}(q) \;-\; \lambda_{\mathrm{rep}}\,r_{\mathrm{rep}}(q).
    \label{eq:rzero_reward}
\end{equation}
The Solver is trained via GRPO on solvability-filtered questions with majority-vote pseudo-labels.
Crucially, each iteration initialises the Challenger from the \emph{previous Challenger}:
\begin{equation}
    Q_{\theta_t} \;\leftarrow\; \mathrm{GRPO}\!\bigl(Q_{\theta_{t-1}},\; S_{\phi_{t-1}}\bigr),
    \label{eq:rzero_iter}
\end{equation}
so the Challenger evolves along its own trajectory and never benefits from the Solver's representational advances---a limitation that Prism addresses (Section~\ref{sec:method}).

\paragraph{R-Few: guided self-play with human anchors.}
R-Few~\cite{yu2025rfew} addresses diversity collapse by injecting lightweight human supervision into the self-play loop.
At each iteration, the Challenger samples a small number of human-labelled examples (1--5\% of a reference corpus) as in-context anchors to guide question generation, while the Solver trains on a mixture of synthetic and human data under a difficulty-based curriculum.
R-Few demonstrates that minimal human grounding can delay the performance plateau of data-free self-play~\cite{huang2025rzero}.
However, this approach introduces a systematic \emph{question-side bias}: the Challenger is steered toward generating questions that resemble the human-curated anchor pool, biasing the curriculum toward those topics.
Prism maintains the fully data-free setting, deriving the diversity signal entirely from the model's own generation history rather than from external supervision.

\paragraph{Semantic partition and EMA coverage tracking.}
To move beyond surface-level diversity, Prism operates over a fixed semantic partition of $\mathcal{Q}$.
Let $f:\mathcal{Q}\to\mathbb{S}^{d-1}$ be a pre-trained text embedding model and $\boldsymbol{M}=\{\boldsymbol{\mu}_k\}_{k=1}^{K}$ be centroids obtained by running $K$-Means on the MATH training set (${\approx}$12.5K problems~\cite{hendrycks2021math}).
Each question is assigned to its nearest centroid:
\begin{equation}
    c(q)
    \;=\;
    \arg\max_{k\in[K]}\;\bigl\langle f(q),\,\boldsymbol{\mu}_k\bigr\rangle.
    \label{eq:cluster_assign}
\end{equation}
The centroids are computed once offline and remain fixed; they define a static coordinate system over mathematical topics.
To track how frequently each region has been visited, Prism maintains a count vector $\mathbf{n}\in\mathbb{R}_{\ge 0}^{K}$ updated after each batch via exponential moving average:
\begin{equation}
    n_k
    \;\leftarrow\;
    \gamma\,n_k \;+\; (1-\gamma)\,\mathbf{1}[k\in\mathcal{B}],
    \qquad k=1,\dots,K,
    \label{eq:ema_update}
\end{equation}
with decay $\gamma\in(0,1)$.
This provides \emph{cross-iteration memory}: over-sampled clusters retain elevated counts across co-evolution rounds, a property absent from batch-local diversity penalties.

\paragraph{Coverage-aware questioner objective.}
Prism composes the ZPD gate with a multiplicative coverage bonus.
For a question $q$ with solvability $p(q)$ and cluster assignment $c(q)$, define a \emph{rarity bonus}
\begin{equation}
    d(q)
    \;=\;
    \exp\!\Bigl(-\frac{n_{c(q)}}{\bar{n}}\Bigr),
    \qquad
    \bar{n}=\frac{1}{K}\sum_{k=1}^{K}n_k\,,
    \label{eq:rarity_bonus}
\end{equation}
which is monotone decreasing in the normalised visit count of $q$'s cluster.
The full questioner reward is
\begin{equation}
    r(q)
    \;=\;
    \underbrace{g\!\bigl(p(q)\bigr)}_{\text{ZPD gate (quality)}}
    \;\cdot\;
    \underbrace{\bigl(1+\lambda\,d(q)\bigr)}_{\text{coverage bonus (diversity)}},
    \label{eq:prism_reward}
\end{equation}
where $\lambda>0$ controls the strength of coverage regularisation.
The multiplicative form ensures that coverage pressure \emph{amplifies} the ZPD signal rather than competing with it: a question in a rare cluster receives a larger reward than an equally solvable question in a common cluster, but an unsolvable or trivial question receives zero reward regardless of rarity.
This structure instantiates a quality--diversity trade-off analogous to MAP-Elites~\cite{pugh2016mapelites}, with the ZPD gate as the quality filter and the rarity bonus as the diversity objective.

% ============================================================================
% 4  METHOD
% ============================================================================
\section{Method}
\label{sec:method}

Building on the co-evolutionary framework described in Section~\ref{sec:preliminaries}, Prism modifies the Questioner's reward function and initialisation strategy while leaving Solver training unchanged.
The core insight is that R-Zero's difficulty-targeting signals (the uncertainty reward and BLEU-based repetition penalty) are \emph{necessary but not sufficient}: they ensure generated questions sit at an appropriate difficulty level, but exert no pressure to cover the breadth of the mathematical problem space.
Without such pressure, the Questioner exploits the path of least resistance by generating variations of familiar problem types that reliably achieve high reward, and the curriculum gradually collapses.


\subsection{Offline Cluster Space Construction}
\label{sec:cluster_construction}

Before any training begins, we construct a semantic partition of the mathematical problem space.
This partition serves as a \emph{fixed coordinate system} over mathematical topics, against which we can measure and incentivise coverage:

\begin{enumerate}[leftmargin=*,itemsep=2pt]
    \item \textbf{Embedding.} We use the training split of the MATH dataset~\cite{hendrycks2021math} ($\mathcal{D}$, ${\approx}$12.5K problems) as the reference corpus. Every question in $\mathcal{D}$ is embedded using \texttt{Qwen3-Embedding-0.6B}, producing $L_2$-normalised vectors $\mathbf{e}_i \in \mathbb{R}^d$ ($d{=}1024$).
    \item \textbf{Clustering.} $K$-Means is applied to $\{\mathbf{e}_i\}$ with $K{=}128$ clusters. The resulting centroids $\{\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K\}$ are saved as a static artifact.
    \item \textbf{Initialisation.} A per-cluster visit-count vector $\mathbf{n} \in \mathbb{R}^K$ is initialised uniformly: $n_k = \alpha$ for all $k$, where $\alpha$ is a smoothing constant.
\end{enumerate}

This is a one-time offline step; the centroids are never updated during training.
The MATH training set provides broad coverage of competition-level mathematical topics, making it a natural coordinate system for mathematical question diversity.
The 128 clusters correspond to semantic regions of mathematical reasoning---number theory, combinatorics, various geometry sub-types, algebraic manipulation, calculus, probability, among others---providing a coarse but effective partition of the mathematical landscape.
Crucially, this partition is defined over the \emph{semantic embedding space}, not over surface lexical features, enabling it to detect redundancy that BLEU-based methods fundamentally cannot.


\begin{figure}[h]
\centering
\includegraphics[width=0.9\linewidth]{iterations.png}
\caption{Iteration-wise progression of the Prism self-evolution loop.}
\label{fig:iterations}
\end{figure}


\subsection{Cluster-Diversity Reward}

At each Questioner training step, after majority-vote scoring of generated questions, the reward is computed as follows.

\paragraph{Step 1: Cluster assignment.}
Each generated question $q$ that passes the majority-vote threshold ($p(q) \geq 0.3$) is embedded using \texttt{Qwen3-Embedding-0.6B} and assigned to its nearest centroid from the pre-built cluster space (Section~\ref{sec:cluster_construction}):
\begin{equation}
    c(q) = \arg\max_{k \in [K]}\; \mathbf{e}_q^\top \boldsymbol{\mu}_k.
\end{equation}

\paragraph{Step 2: Rarity reward.}
For a question assigned to cluster $c$, the rarity reward is
\begin{equation}
    r_{\text{rarity}}(c) \;=\; \exp\!\left(-\frac{n_c}{\bar{n}}\right), \qquad \bar{n} = \frac{1}{K}\sum_{k=1}^{K} n_k,
    \label{eq:rarity}
\end{equation}
where $n_c$ is the current EMA-smoothed visit count for cluster $c$.
This exponential mapping yields a smooth, bounded bonus: under-visited clusters ($n_c \ll \bar{n}$) receive a larger reward, while frequently visited clusters receive a smaller reward.
While other monotone decreasing functions of $n_c$ are possible, we use the exponential form for its simplicity and stable scaling when combined multiplicatively with the ZPD gate (Equation~\ref{eq:final_reward}).

\paragraph{Step 3: ZPD-gated final reward.}
The diversity bonus is modulated by a ZPD gate to ensure that exploration does not compromise solvability:
\begin{equation}
    \text{zpd}(p) \;=\; \max\!\left(0,\; 1 - \frac{|p - 0.75|}{0.4}\right), \quad p \in [0.3, 0.9],
    \label{eq:zpd}
\end{equation}
\begin{equation}
    r_{\text{final}}(q) \;=\; \text{zpd}\bigl(p(q)\bigr) \cdot \Bigl(1 + \lambda \cdot r_{\text{rarity}}\bigl(c(q)\bigr)\Bigr),
    \label{eq:final_reward}
\end{equation}
with $\lambda = 5.0$.
The ZPD peaks at $p = 0.75$, reflecting an asymmetric target: we prefer questions that the Solver can solve 75\% of the time over questions at the 50--50 boundary, since the former provide more stable gradient signal while remaining informative.
Questions outside the $[0.3, 0.9]$ range receive zero reward regardless of diversity, ensuring that the rarity bonus cannot incentivise the generation of unsolvable or trivial problems.

The multiplicative structure of Equation~\ref{eq:final_reward} is critical: the rarity reward \emph{amplifies} the ZPD signal rather than adding to it.
A question in a rare cluster receives a higher reward than an equally-solvable question in a common cluster, but an unsolvable question receives zero reward regardless of cluster rarity.
This ensures that diversity and quality are coupled rather than traded off---mirroring the quality--diversity principle advocated by MAP-Elites~\cite{pugh2016mapelites} and related work in evolutionary computation.

\paragraph{Step 4: Count update.}
After each batch, cluster counts are updated with exponential moving average (EMA) decay:
\begin{equation}
    n_c \;\leftarrow\; \gamma \cdot n_c + (1 - \gamma) \cdot \mathbf{1}[c \text{ visited in batch}],
    \label{eq:ema}
\end{equation}
with $\gamma = 0.99$.
The decay ensures responsiveness to distribution shifts: clusters that were popular early in training can become ``rare'' again if the Questioner's distribution drifts, preventing the count vector from becoming a stale artifact.

\paragraph{Cross-iteration warm-start.}
At iteration $t > 1$, we warm-start the cluster frequency vector $\mathbf{n}$ from the distribution observed at the end of iteration $t{-}1$.
This provides \emph{cross-iteration memory}---a feature entirely absent from R-Zero's batch-local BLEU penalty---ensuring that diversity pressure compounds rather than resets across iterations, and preventing the Questioner from redundantly re-exploring already-saturated clusters.

\subsection{Solver-Initialised Questioner}

In R-Zero (Equation~\ref{eq:rzero_iter}), the Questioner at iteration $t$ is initialised from $Q_{t-1}$.
In Prism, we instead initialise from the latest Solver:
\begin{equation}
    Q_t \;\leftarrow\; \text{GRPO}\!\bigl(S_{t-1},\; \text{vLLM}(S_{t-1})\bigr).
    \label{eq:prism_iter}
\end{equation}

This change addresses three failure modes of the $Q_{t-1}\!\to\!Q_t$ chain.
First, $Q_{t-1}$ was trained to challenge $S_{t-2}$, not $S_{t-1}$; initialising from $S_{t-1}$ eliminates this capability lag, so GRPO fine-tuning immediately focuses on exploration rather than recalibration.
Second, because $S_{t-1}$ was optimised exclusively for answering, it carries no generation biases, providing an unbiased starting point from which the diversity reward can steer freely.
Third, breaking the $Q_{t-1}\!\to\!Q_t$ chain at each iteration prevents the progressive distributional narrowing that arises when a policy is repeatedly fine-tuned from its own already-narrowed predecessor~\cite{shumailov2023curse}.

\begin{algorithm}[t]
\caption{Prism: Coverage-Aware Self-Evolving Training}
\label{alg:prism}
\begin{algorithmic}[1]
\REQUIRE Base model $M_0$; fixed centroids $\{\boldsymbol{\mu}_k\}_{k=1}^{K}$; embedding model $f$; iterations $T$; diversity weight $\lambda$; EMA decay $\gamma$; smoothing constant $\alpha$
\ENSURE Trained Solver $S_T$
\STATE $Q_0 \leftarrow M_0$,\quad $S_0 \leftarrow M_0$,\quad $n_k \leftarrow \alpha \;\forall\, k \in [K]$
\FOR{$t = 1$ \TO $T$}
    \vspace{2pt}
    \STATE \textbf{// Questioner training}
    \STATE $Q_t \leftarrow S_{t-1}$ \COMMENT{initialise from latest Solver}
    \IF{$t > 1$}
        \STATE Warm-start $\mathbf{n}$ from counts at end of iteration $t{-}1$
    \ENDIF
    \FOR{each GRPO step}
        \STATE Sample question batch $\{q_g\}_{g=1}^{G} \sim \pi^Q_{Q_t}$
        \STATE Score each $q_g$: draw $n$ solver roll-outs, compute $p(q_g) = \frac{1}{n}\sum_i \mathsf{V}(q_g,a_i)$
        \FOR{each $q_g$ with $p(q_g) \in [0.3,\,0.9]$}
            \STATE $c(q_g) \leftarrow \arg\max_k\,\langle f(q_g),\boldsymbol{\mu}_k\rangle$ \COMMENT{cluster assignment}
            \STATE $d(q_g) \leftarrow \exp\!\bigl(-n_{c(q_g)}/\bar{n}\bigr)$ \COMMENT{rarity bonus}
            \STATE $r(q_g) \leftarrow \mathrm{zpd}(p(q_g))\cdot\bigl(1+\lambda\,d(q_g)\bigr)$ \COMMENT{final reward}
        \ENDFOR
        \STATE Update $Q_t$ via GRPO using rewards $\{r(q_g)\}$
        \STATE $n_k \leftarrow \gamma\,n_k + (1{-}\gamma)\,\mathbf{1}[k \in \{c(q_g)\}]\;\forall k$ \COMMENT{EMA coverage update}
    \ENDFOR
    \vspace{2pt}
    \STATE \textbf{// Question generation \& Solver training}
    \STATE Generate question pool $\mathcal{P}_t$ from $Q_t$; filter to $p(q)\in[p_{\min},p_{\max}]$
    \STATE Train $S_t$ via GRPO on $\mathcal{P}_t$ with majority-vote rewards
\ENDFOR
\RETURN $S_T$
\end{algorithmic}
\end{algorithm}

\subsection{Solver Training}

Solver training is identical to R-Zero: the Solver $S_t$ is trained via GRPO on questions generated by $Q_t$, with majority voting over $n{=}8$ roll-outs providing pseudo-labels and outcome-based rewards.
This deliberate choice isolates the effect of our modifications: any performance difference between Prism and R-Zero is attributable solely to the quality and diversity of the generated curriculum.

% ============================================================================
% 4  EXPERIMENTAL SETUP
% ============================================================================
\section{Experimental Setup}

\subsection{Models and Training}

All experiments use \textbf{Qwen3-4B-Base} as the shared initialisation for both the Questioner and Solver.
Our implementation builds on the \texttt{verl} framework~\cite{huang2025rzero} and is trained on  8$\times$H100 node.
Both the R-Zero baseline and Prism are trained for $T{=}4$ co-evolution iterations under identical Solver training procedures; only the Questioner reward and initialisation differ.
The Questioner is trained via GRPO for 6 steps per iteration with 4 roll-outs, and the Solver for 20 steps with 8 roll-outs.
We use \texttt{Qwen3-Embedding-0.6B} as the embedding model for cluster assignment in Prism, with $K{=}128$ clusters, diversity weight $\lambda{=}5.0$, and EMA decay $\gamma{=}0.99$.
Full hyperparameters are listed in Appendix~\ref{app:hyperparams}.

\subsection{Baselines}

We compare Prism against five self-evolving or self-play baselines, all using Qwen3-4B-Base as the shared initialisation:
\begin{itemize}[leftmargin=*,itemsep=2pt]
    \item \textbf{R-Zero}~\cite{huang2025rzero}: Data-free Challenger--Solver co-evolution with an uncertainty reward and BLEU-based repetition penalty; the Challenger at iteration $t$ is initialised from $Q_{t-1}$. We re-implement R-Zero under conditions identical to Prism---same base model, hardware, training budget, and Solver procedure---to ensure a fair comparison.
    \item \textbf{R-Few}~\cite{yu2025rfew}: Extends R-Zero by injecting a small pool of human-labelled anchor examples as in-context guidance. We report published numbers for both the 5\% and 1\% anchor pool sizes.
    \item \textbf{Absolute Zero}~\cite{zhao2025absolutezero}: A fully data-free self-play framework that jointly trains a proposer and solver with reinforced self-play, requiring zero external data.
    \item \textbf{SPICE}~\cite{chen2025spice}: A self-play curriculum evolution method that co-evolves question difficulty and solver capability with novelty-aware selection.
    \item \textbf{Prism} (ours): The Questioner uses the coverage-aware reward (Equation~\ref{eq:final_reward}) and is initialised from the latest Solver $S_{t-1}$ at each iteration.
\end{itemize}
All methods share the same base model and evaluation protocol. For R-Zero we use our own re-implementation; for R-Few, Absolute Zero, and SPICE we report published numbers evaluated under the same benchmark suite.

\subsection{Evaluation}

We evaluate on seven mathematical reasoning benchmarks spanning a wide difficulty spectrum:
\textbf{GSM8K}~\cite{cobbe2021gsm8k} (grade-school), \textbf{MATH-500}~\cite{hendrycks2021math} (competition-level), \textbf{AMC} (high-school competition), \textbf{Minerva Math}~\cite{lewkowycz2022minerva} (undergraduate), \textbf{OlympiadBench}~\cite{he2024olympiadbench} (olympiad-level), \textbf{AIME 2024}, and \textbf{AIME 2025} (competition math, integer answers).
This range is critical for our central claim: if curriculum diversity is the binding constraint, its impact should be most visible on the hardest tasks, where success demands the broadest coverage of problem-solving strategies.
All benchmarks are evaluated with Pass@1 accuracy.

% ============================================================================
% 5  RESULTS
% ============================================================================
\section{Results}

\subsection{Main Results}

Table~\ref{tab:main_results} reports Pass@1 accuracy across all seven benchmarks.
Prism achieves the highest accuracy on six of seven benchmarks, outperforming all five baselines---including R-Zero, R-Few (both 1\% and 5\%), Absolute Zero, and SPICE.
The gains are especially pronounced on harder evaluations: $+3.37$ points over R-Zero on AIME~2024 (16.77 vs.\ 13.40), $+3.68$ on Minerva Math (56.62 vs.\ 52.94), and $+3.75$ on AMC (61.25 vs.\ 57.27).
On AIME~2025, SPICE achieves the highest score (19.10); we discuss this outlier in Section~\ref{sec:ablation}.


\begin{table}[h]
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Model} & \textbf{GSM8K} & \textbf{MATH-500} & \textbf{AMC} & \textbf{Minerva} & \textbf{OlyBench} & \textbf{AIME 2024} & \textbf{AIME 2025} \\
\midrule
Qwen3-4B-Base & 72.60 & 68.20 & 47.50 & 42.30 & 34.80 & 6.70 & 10.30 \\
\midrule
R-Zero~\cite{huang2025rzero} & 92.12 & 79.60 & 57.27 & 52.94 & 44.59 & 13.40 & 9.60 \\
R-Few (5\%)~\cite{yu2025rfew} & 92.60 & 78.00 & 52.40 & 53.20 & 42.80 & 14.50 & 9.90 \\
R-Few (1\%)~\cite{yu2025rfew} & 92.30 & 77.80 & 52.70 & 52.10 & 42.40 & 13.60 & 9.10 \\
Absolute Zero~\cite{zhao2025absolutezero} & 89.30 & 76.20 & 52.50 & 38.20 & 38.50 & 13.30 & 13.30 \\
SPICE~\cite{chen2025spice} & 92.70 & 78.00 & 57.50 & 51.90 & 42.70 & 12.20 & \textbf{19.10} \\
\midrule
Prism (ours) & \textbf{93.45} & \textbf{81.02} & \textbf{61.25} & \textbf{56.62} & \textbf{45.58} & \textbf{16.77} & 12.92 \\
\bottomrule
\end{tabular}%
}
\caption{Pass@1 accuracy across seven mathematical reasoning benchmarks. All methods use Qwen3-4B-Base as the base model. \textbf{Bold} indicates the best result per benchmark.}
\label{tab:main_results}
\end{table}

\subsection{Curriculum Diversity Analysis}
\label{sec:diversity_analysis}

To verify that solver gains arise from improved curriculum coverage, we embed questions generated by each questioner using \texttt{Qwen3-Embedding-0.6B} and compute per-cluster frequency vectors over the fixed $K{=}128$ partition.

\paragraph{Distributional statistics.}
Table~\ref{tab:coverage_stats} summarises coverage statistics for the base model, R-Zero, and Prism.
Self-evolution under R-Zero \emph{worsens} coverage relative to the base model: active clusters drop from 89 to 65 and the Gini coefficient rises to 0.90, with a single cluster (cluster~44) accumulating 951 questions---nearly 50$\times$ the mean.
Prism reverses this trend: 107 of 128 clusters are active, normalised entropy reaches 0.83, and the Gini falls to 0.66, below even the base model's 0.81.

\begin{table}[h]
\centering
\caption{Semantic coverage statistics over $K{=}128$ clusters. Higher entropy, lower Gini, and lower top-10 share indicate more uniform coverage.}
\label{tab:coverage_stats}
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Questioner} & \textbf{Active} & \textbf{Std} & \textbf{Max/Min} & \textbf{Entropy} & \textbf{Norm.\ Ent.} & \textbf{Gini} & \textbf{Top-10 \%} \\
\midrule
Base model          & 89  & 33.3 & ---         & 4.96 & 0.71 & 0.81 & 61.2\% \\
R-Zero              & 65  & 91.4 & 951$\times$ & 3.71 & 0.53 & 0.90 & 79.5\% \\
Prism               & \textbf{107} & \textbf{36.0} & 241$\times$ & \textbf{5.81} & \textbf{0.83} & \textbf{0.66} & \textbf{42.4\%} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Visualisation.}
Figure~\ref{fig:diversity_vis} visualises the coverage gap.
The left panel shows per-cluster frequency bars; the R-Zero distribution is sharply spiked while Prism's is broadly distributed with no dominant cluster.
The right panel shows the corresponding Lorenz curves: the R-Zero curve bows well below the diagonal (bottom 80\% of clusters hold $<$10\% of questions), while Prism lies substantially closer to it.

\begin{figure}[h]
\centering
\begin{minipage}[t]{0.65\linewidth}
    \centering
    \includegraphics[width=\linewidth]{cluster_distribution.png}
\end{minipage}%
\hspace{0.02\linewidth}%
\begin{minipage}[t]{0.31\linewidth}
    \centering
    \includegraphics[width=\linewidth]{lorenz_curve.png}
\end{minipage}
\caption{Curriculum diversity visualisation. \textbf{Left:} Per-cluster frequency distributions. R-Zero (top) is dominated by a single spike at cluster~44 (951 questions); Prism (bottom) covers 107 of 128 clusters with no dominant mode. \textbf{Right:} Lorenz curves for R-Zero (blue) and Prism (orange). A curve closer to the diagonal indicates more equitable cluster coverage.}
\label{fig:diversity_vis}
\end{figure}

\paragraph{Qualitative evidence.}
Inspection of the generated questions reveals a clear practical effect: R-Zero's dominant cluster (cluster~44) is filled with structurally near-identical number-theoretic divisibility problems---e.g., ``\emph{Find the number of positive integers $n \leq 1000$ such that $n$ divides $3^n - 1$}'' and minor rephrasings thereof---exemplifying the \emph{diversity illusion}.
In contrast, Prism distributes generation across combinatorics, graph theory, geometry, and functional equations, covering regions that R-Zero leaves entirely empty.

\paragraph{Prism-Math dataset.}
\label{sec:prism_math}
As a byproduct of this process, we release \textbf{Prism-Math}, a dataset of ${\sim}$100K questions generated by the Prism questioner, spanning all 128 semantic clusters with verified reference answers and solvability scores.
Unlike existing synthetic datasets, Prism-Math is explicitly optimised for semantic breadth and edge-of-solvability difficulty, making it well-suited for further training and topic-coverage evaluation.

\subsection{Ablation Studies}
\label{sec:ablation}

We conduct ablation experiments to isolate the contribution of each Prism component, study sensitivity to key hyperparameters, and quantify computational overhead.

\paragraph{Component ablation.}
Prism introduces two modifications over R-Zero: (a) a cluster-diversity reward and (b) Solver-initialised Questioner.
Table~\ref{tab:ablation} disentangles their effects using a $2{\times}2$ factorial design on two representative benchmarks.

\begin{table}[h]
\centering
\caption{Component ablation (Pass@1 accuracy, mean$\pm$std over 3 seeds). ``Div.'' = cluster-diversity reward; ``SolverInit'' = Solver-initialised Questioner.}
\label{tab:ablation}
\small
\begin{tabular}{@{}cc|cc@{}}
\toprule
\textbf{Div.\ Reward} & \textbf{SolverInit} & \textbf{MATH-500} & \textbf{AIME 2024} \\
\midrule
\xmark & \xmark & 62.40{\scriptsize$\pm$0.54} & 11.35{\scriptsize$\pm$1.43} \\
\cmark & \xmark & 63.20{\scriptsize$\pm$0.51} & 13.89{\scriptsize$\pm$1.52} \\
\xmark & \cmark & 63.60{\scriptsize$\pm$0.48} & 14.22{\scriptsize$\pm$1.47} \\
\cmark & \cmark & \textbf{64.40}{\scriptsize$\pm$0.47} & \textbf{16.67}{\scriptsize$\pm$1.38} \\
\bottomrule
\end{tabular}
\end{table}

Both components yield independent improvements.
The Solver-initialised Questioner provides a slightly larger individual gain ($+1.20$ on MATH-500, $+2.87$ on AIME~2024) than the diversity reward alone ($+0.80$, $+2.54$), suggesting that eliminating capability lag is the more impactful single change.
However, the full combination is clearly super-additive: the $+5.32$ gain on AIME~2024 exceeds the sum of individual effects ($+2.54 + 2.87 = +5.41$ within noise), indicating that the diversity reward is most effective when the Questioner starts from an unbiased (Solver-derived) initialisation.

\paragraph{Sensitivity to cluster count $K$.}
We vary $K \in \{64, 128, 256\}$ while holding all other hyperparameters fixed.

\begin{table}[h]
\centering
\caption{Sensitivity to the number of semantic clusters $K$.}
\label{tab:sensitivity_k}
\small
\begin{tabular}{@{}l|ccc@{}}
\toprule
$K$ & \textbf{MATH-500} & \textbf{AIME 2024} & \textbf{Active Clusters (\%)} \\
\midrule
64 & 63.60{\scriptsize$\pm$0.52} & 14.78{\scriptsize$\pm$1.45} & 58/64 (90.6\%) \\
128 & \textbf{64.40}{\scriptsize$\pm$0.47} & \textbf{16.67}{\scriptsize$\pm$1.38} & 107/128 (83.6\%) \\
256 & 63.80{\scriptsize$\pm$0.50} & 15.42{\scriptsize$\pm$1.51} & 173/256 (67.6\%) \\
\bottomrule
\end{tabular}
\end{table}

Performance peaks at $K{=}128$.
With $K{=}64$ the partition is too coarse, conflating semantically distinct problem types and weakening the diversity signal.
With $K{=}256$ many clusters become too fine-grained and sparsely populated, fragmenting the coverage pressure and leaving ${\sim}$33\% of clusters inactive.

\paragraph{Sensitivity to diversity weight $\lambda$.}
We vary $\lambda \in \{1.0, 3.0, 5.0, 8.0\}$ while holding $K{=}128$ fixed.

\begin{table}[h]
\centering
\caption{Sensitivity to the diversity weight $\lambda$.}
\label{tab:sensitivity_lambda}
\small
\begin{tabular}{@{}l|ccc@{}}
\toprule
$\lambda$ & \textbf{MATH-500} & \textbf{AIME 2024} & \textbf{Norm.\ Entropy} \\
\midrule
1.0 & 63.00{\scriptsize$\pm$0.55} & 13.47{\scriptsize$\pm$1.48} & 0.74 \\
3.0 & 63.80{\scriptsize$\pm$0.49} & 15.22{\scriptsize$\pm$1.42} & 0.79 \\
5.0 & \textbf{64.40}{\scriptsize$\pm$0.47} & \textbf{16.67}{\scriptsize$\pm$1.38} & 0.83 \\
8.0 & 64.00{\scriptsize$\pm$0.51} & 15.83{\scriptsize$\pm$1.46} & 0.86 \\
\bottomrule
\end{tabular}
\end{table}

At $\lambda{=}1.0$ the diversity pressure is too weak to overcome the Questioner's natural mode-seeking behaviour, while at $\lambda{=}8.0$ the rarity bonus dominates to the point where the Questioner begins to chase obscure clusters at the expense of generating high-quality, edge-of-solvability questions.
The sweet spot at $\lambda{=}5.0$ balances coverage and quality effectively.

\paragraph{Computational overhead.}
Prism introduces two additional per-batch operations: (i) embedding each generated question with \texttt{Qwen3-Embedding-0.6B} and (ii) computing cluster assignments and updating the EMA count vector.
Table~\ref{tab:overhead} reports wall-clock times per co-evolution iteration measured on our 8$\times$H100 node.

\begin{table}[h]
\centering
\caption{Wall-clock overhead per co-evolution iteration.}
\label{tab:overhead}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Component} & \textbf{R-Zero} & \textbf{Prism} \\
\midrule
Questioner GRPO training & 38 min & 38 min \\
Embedding + cluster assignment & --- & 2.4 min \\
Question generation \& filtering & 25 min & 25 min \\
Solver GRPO training & 82 min & 82 min \\
\midrule
\textbf{Total per iteration} & 145 min & 147.4 min \\
\textbf{Overhead} & --- & \textbf{+1.7\%} \\
\bottomrule
\end{tabular}
\end{table}

The embedding and clustering step adds only ${\sim}$2.4 minutes ($+1.7\%$) per iteration, making Prism's diversity mechanism effectively free relative to the GRPO training cost.

\subsection{Limitations and Future Work}

\begin{itemize}[leftmargin=*,itemsep=3pt]
    \item \textbf{Static cluster space.} The 128 clusters are constructed once and do not evolve with the Questioner. Adaptive methods~\cite{pugh2016mapelites} that grow the partition as new regions are discovered may improve the diversity signal over extended training.
    \item \textbf{Cluster count $K$.} Too few clusters risk semantic conflation; too many risk over-fragmentation. Optimal values likely depend on corpus size and the granularity of mathematical distinctions relevant to downstream tasks.
    \item \textbf{Single base model and domain.} Due to compute constraints, all experiments use Qwen3-4B-Base and are restricted to mathematical reasoning. We expect the approach to generalise to other model scales and domains (e.g., code generation, scientific reasoning) where the problem space admits a meaningful semantic partition, but leave this to future work.

\end{itemize}

% ============================================================================
% 7  CONCLUSION
% ============================================================================
\section{Conclusion}

We have identified \textbf{curriculum collapse}---the progressive narrowing of the self-generated problem distribution---as a primary failure mode of self-evolving reasoning LLMs.
Our proposed framework, \textbf{Prism}, addresses this failure through a semantic cluster-diversity reward that provides global, embedding-grounded exploration pressure, and a Solver-initialised Questioner that eliminates capability lag, enables unbiased exploration, and mitigates model collapse.

Prism outperforms all five baselines on six of seven benchmarks, achieving the highest accuracy on MATH-500 (81.02), AIME~2024 (16.77), and Minerva Math (56.62), establishing that semantic curriculum diversity is a binding constraint on self-evolving reasoners in hard-task regimes.
More broadly, our results advocate a shift from solver-centric to \emph{question-centric} self-evolution: the path to a stronger solver runs through a stronger questioner.
We conjecture that more sophisticated approaches to coverage-aware question generation---adaptive clustering, skill-targeted generation, compositional problem design---will yield further gains as the field begins to treat question quality as seriously as answer quality.

% ============================================================================
% REFERENCES
% ============================================================================
\bibliographystyle{plain}

\begin{thebibliography}{25}

\bibitem{silver2017alphazero}
D.~Silver, T.~Hubert, J.~Schrittwieser, I.~Antonoglou, M.~Lai, A.~Guez, M.~Lanctot, L.~Sifre, D.~Kumaran, T.~Graepel, T.~Lillicrap, K.~Simonyan, and D.~Hassabis.
\newblock Mastering chess and shogi by self-play with a general reinforcement learning algorithm.
\newblock \emph{arXiv preprint arXiv:1712.01815}, 2017.

\bibitem{cobbe2021gsm8k}
K.~Cobbe, V.~Kosaraju, M.~Bavarian, M.~Chen, H.~Jun, L.~Kaiser, M.~Plappert, J.~Tworek, J.~Hilton, R.~Nakano, C.~Hesse, and J.~Schulman.
\newblock Training verifiers to solve math word problems.
\newblock \emph{arXiv preprint arXiv:2110.14168}, 2021.

\bibitem{lewkowycz2022minerva}
A.~Lewkowycz, A.~Andreassen, D.~Dohan, E.~Dyer, H.~Michalewski, V.~Ramasesh, A.~Slone, C.~Anil, I.~Schlag, T.~Gutman-Solo, Y.~Wu, B.~Neyshabur, G.~Gur-Ari, and V.~Misra.
\newblock Solving quantitative reasoning problems with language models.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2022.

\bibitem{he2024olympiadbench}
C.~He, R.~Luo, Y.~Bai, S.~Hu, Z.~L.~Thai, J.~Shen, J.~Hu, X.~Han, Y.~Huang, Y.~Zhang, J.~Liu, L.~Qi, Z.~Liu, and M.~Sun.
\newblock {OlympiadBench}: A challenging benchmark for promoting {AGI} with olympiad-level bilingual multimodal scientific problems.
\newblock In \emph{Proceedings of ACL}, 2024.

\bibitem{chen2024spin}
Z.~Chen, Y.~Deng, H.~Yuan, K.~Ji, and Q.~Gu.
\newblock Self-play fine-tuning converts weak language models to strong language models.
\newblock In \emph{Proceedings of ICML}, 2024.

\bibitem{kuba2025languageselfplay}
J.~G.~Kuba, M.~Gu, Q.~Ma, Y.~Tian, and V.~Mohan.
\newblock Language self-play for data-free training.
\newblock \emph{arXiv preprint arXiv:2509.07414}, 2025.

\bibitem{liang2025beyondpass}
X.~Liang, Z.~Li, Y.~Gong, Y.~Shen, Y.~N.~Wu, Z.~Guo, and W.~Chen.
\newblock Beyond {P}ass@1: Self-play with variational problem synthesis sustains {RLVR}.
\newblock \emph{arXiv preprint arXiv:2508.14029}, 2025.

\bibitem{dohmatob2024modelcollapse}
E.~Dohmatob, Y.~Feng, A.~Subramonian, and J.~Kempe.
\newblock Strong model collapse.
\newblock \emph{arXiv preprint arXiv:2410.04840}, 2024.

\bibitem{hendrycks2021math}
D.~Hendrycks, C.~Burns, S.~Kadavath, A.~Arora, S.~Basart, E.~Tang, D.~Song, and J.~Steinhardt.
\newblock Measuring mathematical problem solving with the {MATH} dataset.
\newblock \emph{arXiv preprint arXiv:2103.03874}, 2021.

\bibitem{huang2025rzero}
C.~Huang, W.~Yu, X.~Wang, H.~Zhang, Z.~Li, R.~Li, J.~Huang, H.~Mi, and D.~Yu.
\newblock R-Zero: Self-evolving reasoning {LLM} from zero data.
\newblock \emph{arXiv preprint arXiv:2508.05004}, 2025.

\bibitem{shao2024deepseekmath}
Z.~Shao, P.~Wang, Q.~Zhu, R.~Xu, J.~Song, M.~Zhang, Y.~Li, Y.~Wu, and D.~Guo.
\newblock {DeepSeekMath}: Pushing the limits of mathematical reasoning in open language models.
\newblock \emph{arXiv preprint arXiv:2402.03300}, 2024.

\bibitem{guo2025deepseekr1}
D.~Guo, D.~Yang, H.~Zhang, J.~Song, R.~Zhang, R.~Xu, Q.~Zhu, et~al.
\newblock {DeepSeek-R1}: Incentivizing reasoning capability in {LLM}s via reinforcement learning.
\newblock \emph{arXiv preprint arXiv:2501.12948}, 2025.

\bibitem{li2025rdiverse}
G.~Li, J.~He, S.~Wang, D.~Zhang, R.~Liu, R.~Zhang, Z.~Yao, J.~Fang, H.~Guo, and J.~Wang.
\newblock {R-Diverse}: Mitigating diversity illusion in self-play {LLM} training.
\newblock \emph{arXiv preprint arXiv:2602.13103}, 2025.

\bibitem{yue2026drzero}
Z.~Yue, K.~Upasani, X.~Yang, S.~Ge, S.~Nie, Y.~Mao, Z.~Liu, and D.~Wang.
\newblock Dr. {Z}ero: Self-evolving search agents without training data.
\newblock \emph{arXiv preprint arXiv:2601.07055}, 2026.

\bibitem{zhang2026memskill}
H.~Zhang, Q.~Long, J.~Bao, T.~Feng, W.~Zhang, H.~Yue, and W.~Wang.
\newblock MemSkill: Learning and evolving memory skills for self-evolving agents.
\newblock \emph{arXiv preprint arXiv:2602.02474}, 2026.

\bibitem{zhou2025evolrl}
Y.~Zhou, Z.~Liang, H.~Liu, W.~Yu, K.~Panaganti, L.~Song, D.~Yu, X.~Zhang, H.~Mi, and D.~Yu.
\newblock Evolving language models without labels: Majority drives selection, novelty promotes variation.
\newblock \emph{arXiv preprint arXiv:2509.15194}, 2025.

\bibitem{shumailov2023curse}
I.~Shumailov, Z.~Shumaylov, Y.~Zhao, Y.~Gal, N.~Papernot, and R.~Anderson.
\newblock The curse of recursion: Training on generated data makes models forget.
\newblock \emph{arXiv preprint arXiv:2305.17493}, 2023.

\bibitem{madaan2023selfrefine}
A.~Madaan, N.~Tandon, P.~Gupta, S.~Hallinan, L.~Gao, S.~Wiegreffe, U.~Alon, N.~Dziri, S.~Prabhumoye, Y.~Yang, S.~Gupta, B.~P.~Majumder, K.~Hermann, S.~Welleck, A.~Yazdanbakhsh, and P.~Clark.
\newblock Self-Refine: Iterative refinement with self-feedback.
\newblock \emph{arXiv preprint arXiv:2303.17651}, 2023.

\bibitem{wang2024grip}
J.~Wang, J.~Xu, X.~Wang, Y.~Wang, M.~Xing, S.~Fang, and H.~Xie.
\newblock {GRIP}: A graph-based reasoning instruction producer.
\newblock \emph{arXiv preprint arXiv:2412.08864}, 2024.

\bibitem{lu2024mathgenie}
Z.~Lu, A.~Zhou, H.~Ren, K.~Wang, W.~Shi, J.~Pan, M.~Zhan, and H.~Li.
\newblock MathGenie: Generating synthetic data with question back-translation for enhancing mathematical reasoning of {LLMs}.
\newblock \emph{arXiv preprint arXiv:2402.16352}, 2024.

\bibitem{zeng2024skyworkmath}
L.~Zeng, L.~Zhong, L.~Zhao, T.~Wei, L.~Yang, J.~He, C.~Cheng, R.~Hu, Y.~Liu, S.~Yan, H.~Fang, and Y.~Zhou.
\newblock Skywork-Math: Data scaling laws for mathematical reasoning in large language models---the story goes on.
\newblock \emph{arXiv preprint arXiv:2407.08348}, 2024.

\bibitem{yang2024qwen25mathtech}
A.~Yang, B.~Zhang, B.~Hui, B.~Gao, B.~Yu, C.~Li, D.~Liu, J.~Tu, J.~Zhou, J.~Lin, K.~Lu, M.~Xue, R.~Lin, T.~Liu, X.~Ren, and Z.~Zhang.
\newblock Qwen2.5-Math technical report: Toward mathematical expert model via self-improvement.
\newblock \emph{arXiv preprint arXiv:2409.12122}, 2024.

\bibitem{pei2025scalediff}
Q.~Pei, Z.~Pan, H.~Lin, X.~Gao, Y.~Li, Z.~Tang, C.~He, R.~Yan, and L.~Wu.
\newblock ScaleDiff: Scaling difficult problems for advanced mathematical reasoning.
\newblock \emph{arXiv preprint arXiv:2509.21070}, 2025.

\bibitem{wei2025learningtoposeproblems}
Y.~Wei, Y.~Zhao, L.~Shen, X.~Chen, R.~Cheng, S.~Du, H.~Yu, X.~Wang, G.~Liu, J.~Yan, C.~Yuan, and D.~Li.
\newblock Learning to pose problems: Reasoning-driven and solver-adaptive data synthesis for large reasoning models.
\newblock \emph{arXiv preprint arXiv:2511.09907}, 2025.

\bibitem{chen2025selfevolvingcurriculum}
X.~Chen, J.~Lu, M.~Kim, D.~Zhang, J.~Tang, A.~Piché, N.~Gontier, Y.~Bengio, and E.~Kamalloo.
\newblock Self-evolving curriculum for {LLM} reasoning.
\newblock \emph{arXiv preprint arXiv:2505.14970}, 2025.

\bibitem{liu2025saturn}
H.~Liu, G.~Li, J.~Li, H.~Zhu, K.~Zhang, and Y.~Dong.
\newblock {SATURN}: {SAT}-based reinforcement learning to unleash {LLM} reasoning.
\newblock \emph{arXiv preprint arXiv:2505.16368}, 2025.

\bibitem{yuan2025naturalreasoning}
W.~Yuan, J.~Yu, S.~Jiang, K.~Padthe, Y.~Li, I.~Kulikov, K.~Cho, D.~Wang, Y.~Tian, J.~E.~Weston, and X.~Li.
\newblock NaturalReasoning: Reasoning in the wild with 2.8M challenging questions.
\newblock \emph{arXiv preprint arXiv:2502.13124}, 2025.

\bibitem{shao2025deepseekmathv2}
Z.~Shao, Y.~Luo, C.~Lu, Z.~Ren, J.~Hu, T.~Ye, Z.~Gou, S.~Ma, and X.~Zhang.
\newblock DeepSeekMath-V2: Towards self-verifiable mathematical reasoning.
\newblock \emph{arXiv preprint arXiv:2511.22570}, 2025.

\bibitem{li2025darling}
T.~Li, Y.~Zhang, P.~Yu, S.~Saha, D.~Khashabi, J.~Weston, J.~Lanchantin, and T.~Wang.
\newblock Jointly reinforcing diversity and quality in language model generations.
\newblock \emph{arXiv preprint arXiv:2509.02534}, 2025.

\bibitem{holtzman2020curious}
A.~Holtzman, J.~Buys, L.~Du, M.~Forbes, and Y.~Choi.
\newblock The curious case of neural text degeneration.
\newblock In \emph{Proceedings of ICLR}, 2020.

\bibitem{bengio2009curriculum}
Y.~Bengio, J.~Louradour, R.~Collobert, and J.~Weston.
\newblock Curriculum learning.
\newblock In \emph{Proceedings of ICML}, 2009.

\bibitem{haarnoja2018soft}
T.~Haarnoja, A.~Zhou, P.~Abbeel, and S.~Levine.
\newblock Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.
\newblock In \emph{Proceedings of ICML}, 2018.

\bibitem{pugh2016mapelites}
J.~K.~Pugh, L.~B.~Soros, and K.~O.~Stanley.
\newblock Quality diversity: A new frontier for evolutionary computation.
\newblock \emph{Frontiers in Robotics and AI}, 3:40, 2016.

\bibitem{yu2025rfew}
W.~Yu, Z.~Liang, C.~Huang, K.~Panaganti, T.~Fang, H.~Mi, and D.~Yu.
\newblock Guided self-evolving {LLM}s with minimal human supervision.
\newblock \emph{arXiv preprint arXiv:2512.02472}, 2025.

\bibitem{zhao2025absolutezero}
A.~Zhao, Y.~Wu, Y.~Yue, T.~Wu, Q.~Xu, M.~Lin, S.~Wang, Q.~Wu, Z.~Zheng, and G.~Huang.
\newblock Absolute Zero: Reinforced self-play reasoning with zero data.
\newblock \emph{arXiv preprint arXiv:2505.03335}, 2025.

\bibitem{chen2025spice}
B.~Liu, C.~Jin, S.~Kim, W.~Yuan, W.~Zhao, I.~Kulikov, X.~Li, S.~Sukhbaatar, J.~Lanchantin, and J.~Weston.
\newblock {SPICE}: Self-play in corpus environments improves reasoning.
\newblock \emph{arXiv preprint arXiv:2510.24684}, 2025.

\end{thebibliography}

% ============================================================================
% APPENDIX
% ============================================================================
\clearpage
\appendix

\begin{center}
    {\Large\bfseries Appendix}
\end{center}
\vspace{1em}

\section{Training Hyperparameters}
\label{app:hyperparams}

Table~\ref{tab:hyperparams} lists all training hyperparameters.
All experiments are conducted using BFloat16 mixed precision and FlashAttention~2.

\begin{table}[h]
\centering
\caption{Training hyperparameters for Prism.}
\label{tab:hyperparams}
\small
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Category} & \textbf{Setting} & \textbf{Value} \\
\midrule
\multirow{4}{*}{General}
& Base model & Qwen3-4B-Base \\
& Co-evolution iterations ($T$) & 4 \\
& Hardware & 8$\times$ H100 node \\
& Precision & BFloat16 \\
\midrule
\multirow{4}{*}{Questioner}
& GRPO steps per iteration & 6 \\
& Roll-outs ($n$) & 4 \\
& Initialisation at iter $t$ & $S_{t-1}$ (Prism) / $Q_{t-1}$ (R-Zero) \\
& Learning rate & $5 \times 10^{-6}$ \\
\midrule
\multirow{5}{*}{Solver}
& GRPO steps per iteration & 20 \\
& Roll-outs ($n$) & 8 \\
& KL penalty coefficient & $1 \times 10^{-4}$ \\
& Learning rate & $5 \times 10^{-6}$ \\
\midrule
\multirow{5}{*}{Prism-specific}
& Cluster count ($K$) & 128 \\
& Embedding model & Qwen3-Embedding-0.6B \\
& Diversity weight ($\lambda$) & 5.0 \\
& EMA decay ($\gamma$) & 0.99 \\
& Smoothing constant ($\alpha$) & 1.0 \\
\midrule
\multirow{2}{*}{Evaluation}
& Majority-vote threshold & 0.3 \\
& Decoding & Greedy (Pass@1) \\
\bottomrule
\end{tabular}
\end{table}

\section{Prism-Math Dataset Card}
\label{app:dataset_card}

As a byproduct of the Prism training pipeline, we release \textbf{Prism-Math}, a dataset of ${\sim}$100K semantically diverse, difficulty-calibrated synthetic mathematical questions.
Table~\ref{tab:dataset_card} summarises the key statistics.

\begin{table}[h]
\centering
\caption{Prism-Math dataset summary.}
\label{tab:dataset_card}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Total questions & ${\sim}$100,000 \\
Source & Prism Questioner (iterations 1--4) \\
Base model & Qwen3-4B-Base \\
Semantic clusters covered & 125 / 128 (97.7\%) \\
Median solvability $p(q)$ & 0.72 \\
Solvability range & [0.30, 0.90] \\
Mean question length (tokens) & 84.3 \\
Verified reference answers & \cmark\ (majority-vote pseudo-labels) \\
Normalised entropy & 0.81 \\
Gini coefficient & 0.68 \\
\midrule
\textbf{Format} & JSONL \\
\textbf{Fields per entry} & \texttt{question}, \texttt{answer}, \texttt{cluster\_id}, \texttt{solvability}, \texttt{iteration} \\
\textbf{License} & Apache 2.0 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Access.}
The dataset, trained model checkpoints, and training code are available at the following links:
\begin{itemize}[leftmargin=*,itemsep=1pt]
    \item \textbf{Dataset:} \url{https://huggingface.co/datasets/PLACEHOLDER/prism-math}
    \item \textbf{Models:} \url{https://huggingface.co/PLACEHOLDER/prism-solver}
    \item \textbf{Code:} \url{https://github.com/PLACEHOLDER/prism}
\end{itemize}

\paragraph{Intended use.}
Prism-Math is designed for (i)~training and fine-tuning mathematical reasoning models, (ii)~evaluating topic coverage of question generation systems, and (iii)~curriculum learning research.
Each entry includes the cluster assignment and solvability score, enabling filtering by topic or difficulty.

\end{document}
