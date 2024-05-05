<h1 align = "center">🤖 Awesome LLM for APR</h1>
<p align="center">
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"></a>
  <a href="https://arxiv.org/abs/2405.01466"><img src="https://img.shields.io/badge/arXiv-2405.01466-blue.svg"></a>
  <img src="https://img.shields.io/github/stars/iSEngLab/AwesomeLLM4APR?color=yellow&label=Stars">
  <img src="https://img.shields.io/badge/PRs-Welcome-red">
  <img src="https://img.shields.io/github/last-commit/iSEngLab/AwesomeLLM4APR">
</p>


## Contents

- [Survey Methodology](#survey-methodology)
- [Publication Trends](#publication-trends)
- [Popular LLMs for APR](#popular-llms-for-apr)
- [LLM Utilization Strategies](#llm-utilization-strategies)
- [Repair Scenarios](#repair-scenarios)
- [Other Critical Aspects](#other-critical-aspects)  
- [Challenges and Future Directions](#challenges-and-future-directions)
- [Citation](#citation)


## Citation

```bibtex
@article{zhang2024survey,
  title={A Systematic Literature Review on Large Language Models for Automated Program Repair},
  author={Zhang, Quanjun and Fang, Chunrong and Xie, Yang and Ma, Yuxiang and Sun, Weisong and Yang, Yun and Chen, Zhenyu},
  journal={arXiv preprint arXiv:2405.01466}
  year={2024}
}

```

## LLM4APR

### CCF A

1. Evaluating representation learning of code changes for predicting patch correctness in program repair [2020-ASE] [[paper](https://arxiv.org/pdf/2008.02944)] [[repo](https://github.com/TruX-DTF/DL4PatchCorrectness)]
1. Generating Bug-Fixes Using Pretrained Transformers [2021-PLDI] [[paper](https://arxiv.org/pdf/2104.07896)]
1. CURE Code-Aware Neural Machine Translation for Automatic Program Repair [2021-ICSE] [[paper](https://arxiv.org/pdf/2103.00073)] [[repo](https://github.com/lin-tan/CURE)]
1. Is this Change the Answer to that Problem? Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness [2021-ASE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3551349.3556914)] [[repo](https://github.com/Trustworthy-Software/Quatrain)]
1. VulRepair: A T5-Based Automated Software Vulnerability Repair [2022-FSE/ESEC] [[paper](https://www.researchgate.net/profile/Chakkrit-Tantithamthavorn/publication/362092639_VulRepair_A_T5-Based_Automated_Software_Vulnerability_Repair/links/6345ea1076e39959d6b73228/VulRepair-A-T5-Based-Automated-Software-Vulnerability-Repair.pdf)] [[repo](https://github.com/awsm-research/VulRepair)]
1. CIRCLE: Continual Repair across Programming Languages [2022-ISSTA] [[paper](https://arxiv.org/pdf/2205.10956)] [[repo](https://github.com/2022CIRCLE/CIRCLE)]
1. An empirical study of deep transfer learning-based program repair for Kotlin projects [2022-FSE/ESEC] [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3558967)]
1. DeepDev-PERF: A Deep Learning-Based Approach for Improving Software Performance [2022-FSE/ESEC] [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549096)] [[repo](https://github.com/glGarg/DeepDev-PERF)]
1. SYNSHINE: improved fixing of Syntax Errors [2022-TSE] [[paper](https://ieeexplore.ieee.org/abstract/document/9913705)] [[repo](https://zenodo.org/record/4572390#.Y4CY8xRByUk)]
1. Less training, more repairing please: revisiting automated program repair via zero-shot learning [2022-FSE/ESEC] [[paer](https://dl.acm.org/doi/pdf/10.1145/3540250.3549101)] [[repo](https://zenodo.org/records/6819444)]
1. DEAR A Novel Deep Learning-based Approach for Automated Program Repair [2022-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3510003.3510177)] [[repo](https://github.com/AutomatedProgramRepair-2021/dear-auto-fix)]
1. Automated Program Repair in the Era of Large Pre-trained Language Models [2023-ICSE] [[paper](https://web.eecs.umich.edu/~movaghar/IEEE-2023-Automated_Program_Repair_in_the_Era_of_Large_Pre-trained_Language_Models.pdf)] [[repo](https://zenodo.org/records/7592886)]
1. Repair Is Nearly Generation: Multilingual Program Repair with LLMs [2023-AAAI] [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/25642/25414)]
1. Retrieval-based prompt selection for code-related few-shot learning [2023-ICSE] [[paper](https://nashid.github.io/resources/papers/cedar-icse23.pdf)] [[repo](https://github.com/prompt-learning/cedar)]
1. What makes good in-context demonstrations for code intelligence tasks with llms? [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298329)] [[repo](https://github.com/shuzhenggao/ICL4code)]
1. Examining zero-shot vulnerability repair with large language models [2023-S&P] [[paper](https://arxiv.org/pdf/2112.02125)] [[repo](https://drive.google.com/drive/folders/1xJ-z2Wvvg7JSaxfTQdxayXFEmoF3y0ET?usp=sharing)]
1. An Empirical Study on Fine-Tuning Large Language Models of Code for Automated Program Repair [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298532)] [[repo](https://github.com/LLMC-APR/STUDY)]
1. Impact of Code Language Models on Automated Program Repair [2023-ICSE] [[paper](https://arxiv.org/pdf/2302.05020)] [[repo](https://github.com/lin-tan/clm)]
1. The Plastic Surgery Hypothesis in the Era of Large Language Models [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298499)] [[repo](https://zenodo.org/records/8244813)]
1. How Effective Are Neural Networks for Fixing Security Vulnerabilities [2023-ISSTA] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597926.3598135)] [[repo](https://github.com/lin-tan/llm-vul)]
1. Pre-trained Model-based Automated Software Vulnerability Repair: How Far are We? [2023-TDSC] [[paper](https://arxiv.org/pdf/2308.12533)] [[repo](https://github.com/iSEngLab/LLM4VulFix)]
1. RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.06057)] [[repo](https://figshare.com/s/a4e95baee01bba14bf4b)]
1. InferFix: End-to-End Program Repair with LLMs over Retrieval-Augmented Prompts [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2303.07263)] [[repo](https://github.com/microsoft/InferredBugs)]
1. Automated Repair of Programs from Large Language Models [2023-ICSE] [[paper](https://arxiv.org/pdf/2205.10583)] [[repo](https://github.com/zhiyufan/apr4codex)]
1. Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.00608)] [[repo](https://github.com/ise-uiuc/Repilot)]
1. Gamma: Revisiting Template-Based Automated Program Repair Via Mask Prediction [2023-ASE] [[paper](https://arxiv.org/pdf/2309.09308)] [[repo](https://github.com/iSEngLab/GAMMA)]
1. Vision Transformer-Inspired Automated Vulnerability Repair [2023-TOSEM] [[paper](https://www.researchgate.net/profile/Michael-Fu-8/publication/375618720_Vision_Transformer-Inspired_Automated_Vulnerability_Repair/links/65a9b875ee1e1951fbbe6538/Vision-Transformer-Inspired-Automated-Vulnerability-Repair.pdf)] [[repo](https://github.com/awsm-research/VQM)]
1. Improving Automated Program Repair with Domain Adaptation [2023-TOSEM] [[paper](https://arxiv.org/pdf/2212.11414)] [[repo](https://github.com/arminzirak/TFix)]
1. Refining ChatGPT-Generated Code: Characterizing and Mitigating Code Quality Issues [2023-TOSEM] [[paper](https://dl.acm.org/doi/pdf/10.1145/3643674)] [[repo](https://github.com/yueyueL/ChatGPT-CodeGenAnalysis)]
1. Self-Refine: Iterative Refinement with Self-Feedback [2023-NeurIPS] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf)] [[repo](https://github.com/madaan/self-refine)]
1. Self-Edit: Fault-Aware Code Editor for Code Generation [2023-ACL] [[paper](https://arxiv.org/pdf/2305.04087)] [[repo](https://github.com/zkcpku/Self-Edit)]
1. Baldur: Whole-Proof Generation and Repair with Large Language Models [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2303.04910)]
1. The Best of Both Worlds: Combining Learned Embeddings with Engineered Features for Accurate Prediction of Correct Patches [2023-TOSME] [[paper](https://dl.acm.org/doi/pdf/10.1145/3576039)] [[repo](https://github.com/HaoyeTianCoder/Panther)]
1. Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning [2023-TSE] [[paper](https://arxiv.org/pdf/2301.01113)] [[repo](https://github.com/thanhlecongg/Invalidator)]
1. Frustrated with Code Quality Issues? LLMs can Help! [2024-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.12938)] [[repo](https://aka.ms/CORE_MSRI)]
1. Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors [2024-ICSE] [[paper](https://arxiv.org/pdf/2306.01394)] [[repo](https://github.com/JohnnyPeng18/TypeFix)]
1. PyTy: Repairing Static Type Errors in Python [2024-ICSE] [[paper](https://arxiv.org/pdf/2401.06619)] [[repo](https://github.com/sola-st/PyTy)]
1. Out of Context: How important is Local Context in Neural Program Repair? [2024-ICSE] [[paper](https://arxiv.org/pdf/2312.04986)] [[repo](https://github.com/giganticode/out_of_context_paper_data)]
1. Out of Sight, Out of Mind: Better Automatic Vulnerability Repair by Broadening Input Ranges and Sources [2024-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3639222)] [[repo](https://github.com/soarsmu/VulMaster_)]
1. Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code [2024-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3639226)] [[repo](https://github.com/Intelligent-CAT-Lab/PLTranslationEmpirical)]
1. Exploring Experiences with Automated Program Repair in Practice [2024-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3639182)]
1. CYCLE: Learning to Self-Refine the Code Generation [2024-OOPSLA] [[paper](https://arxiv.org/pdf/2403.18746)] [[repo](https://github.com/ARiSE-Lab/CYCLE_OOPSLA_24)]
1. Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study [2024-ICSE] [[paper](https://arxiv.org/pdf/2309.08221)] [[repo](https://sites.google.com/view/chatgptcodereview)]
1. APPT Boosting Automated Patch Correctness Prediction via Pre-trained Language Model [2024-TSE] [[paper](https://arxiv.org/pdf/2301.12453)] [[repo](https://github.com/iSEngLab/APPT)]
1. A Deep Dive into Large Language Models for Automated Bug Localization and Repair [2024-FSE/ESEC] [[paper](https://arxiv.org/pdf/2404.11595)]

### CCF B

1. Enhancing Code Language Models for Program Repair by Curricular Fine-tuning Framework [2023-ICSME] [[paper](https://ieeexplore.ieee.org/abstract/document/10336339)]

### CCF C

1. Applying CodeBERT for Automated Program Repair of Java Simple Bugs [2021-MSR] [[paper](https://arxiv.org/pdf/2103.11626)] [[repo](https://github.com/EhsanMashhadi/MSR2021-ProgramRepair)]
2. Evaluating ChatGPT for Smart Contracts Vulnerability Correction [2023-COMPSAC] [[paper](https://ieeexplore.ieee.org/abstract/document/10197134)] [[repo](https://github.com/enaples/solgpt)]
3. Fully Autonomous Programming with Large Language Models [2023-GECCO] [[paper](https://dl.acm.org/doi/pdf/10.1145/3583131.3590481)] [[repo](https://github.com/KoutchemeCharles/aied2023)]

### OTHER

1. TFix-Learning to Fix Coding Errors with a Text-to-Text Transformer [2021-PMLR] [[paper](http://proceedings.mlr.press/v139/berabi21a/berabi21a.pdf)] [[repo](https://github.com/eth-sri/TFix)]
2. Towards JavaScript program repair with Generative Pre-trained Transformer (GPT-2) [2022-APR] [[paper](http://publicatio.bibl.u-szeged.hu/25241/1/Lajko-APR22.pdf)] [[repo](https://github.com/AAI-USZ/APR22-JS-GPT)]
3. Repairing bugs in python assignments using large language models [2022-arixv] [[paper](https://arxiv.org/pdf/2209.14876)]
4. Fix Bugs with Transformer through a Neural-Symbolic Edit Grammar [2022-ICLR] [[paper](https://arxiv.org/pdf/2204.06643)]
5. Patch Generation with Language Models: Feasibility and Scaling Behavior [2022-ICLR] [[paper](https://openreview.net/pdf?id=rHlzJh_b1-5)]
6. Can OpenAI's codex fix bugs?: an evaluation on QuixBugs [2022-APR] [[paper](https://dl.acm.org/doi/abs/10.1145/3524459.3527351)]
7. An Analysis of the Automatic Bug Fixing Performance of ChatGPT [2022-APR] [[paper](https://ieeexplore.ieee.org/abstract/document/10189263)] [[repo](https://gitlab.rlp.net/dsobania/chatgpt-apr)]
8. Framing Program Repair as Code Completion [2022-APR] [[paper](http://repositorio.inesctec.pt/bitstreams/2fb3b152-a3ba-4561-ad11-3869b0d245a0/download)] [[repo](https://github.com/FranciscoRibeiro/code-truncater)]
9. Automated Program Repair Using Generative Models for Code Infilling [2023-AIED] [[paper](https://link.springer.com/chapter/10.1007/978-3-031-36272-9_74)] [[repo](https://github.com/KoutchemeCharles/aied2023)]
10. Fixing Rust Compilation Errors using LLMs [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.05177)]
11. STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.14460)]
12. Conversational automated program repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2301.13246)]
13. Is ChatGPT the Ultimate Programming Assistant--How far is it? [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.11938)] [[repo](https://github.com/HaoyeTianCoder/ChatGPT-Study)]
14. Using Large Language Models for Bug Localization and Fixing [2023-iCAST] [[paper](https://u-aizu.ac.jp/~markov/pubs/iCAST_23.pdf)]
15. SkipAnalyzer: An Embodied Agent for Code Analysis with Large Language Models [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.18532)] [[repo](https://zenodo.org/records/10043170)]
16. An Evaluation of the Effectiveness of OpenAI's ChatGPT for Automated Python Program Bug Fixing using QuixBugs [2023-iSEMANTIC] [[paper](https://ieeexplore.ieee.org/abstract/document/10295323)]
17. Keep the Conversation Going: Fixing 162 out of 337 bugs for $0.42 each using ChatGPT [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.00385)]
18. Explainable Automated Debugging via Large Language Model-driven Scientific Debugging [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.02195)]
19. A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification [2023-arxiv] [[paper](https://arxiv.org/pdf/2305.14752)]
20. A Chain of AI-based Solutions for Resolving FQNs and Fixing Syntax Errors in Partial Code [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.11981)] [[repo](https://github.com/SE-qinghuang/A-Chain-of-AI-based-Solutions-for-Resolving-FQNs-and-Fixing-Syntax-Errors-in-Partial-Code)]
21. GPT-3-Powered Type Error Debugging: Investigating the Use of Large Language Models for Code Repair [2023-SLE] [[paper](https://dl.acm.org/doi/abs/10.1145/3623476.3623522)] [[repo](https://gitlab.com/FranciscoRibeiro/mentat)]
22. Resolving Crash Bugs via Large Language Models: An Empirical Study [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.10448)] [[repo](https://chatgpt4cradiag.github.io/)]
23. The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.17485)]
24. On Hardware Security Bug Code Fixes By Prompting Large Language Models [2023-TIFS] [[paper](https://ieeexplore.ieee.org/abstract/document/10462177)] [[repo](https://zenodo.org/records/10416865)]
25. Towards Generating Functionally Correct Code Edits from Natural Language Issue Descriptions [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.03816)]
26. Exploring the Limits of ChatGPT in Software Security Applications [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.05275)]
27. ZeroLeak: Using LLMs for Scalable and Cost Effective Side-Channel Patching [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.13062)]
28. CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.08588)] [[repo](https://github.com/WeixiangYAN/CodeScope)]
29. How ChatGPT is Solving Vulnerability Management Problem [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.06530)] [[repo](https://anonymous.4open.science/r/DefectManagementEvaluation-0411)]
30. Guiding ChatGPT to Fix Web UI Tests via Explanation-Consistency Checking [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.05778)]
31. Enhancing Automated Program Repair through Fine-tuning and Prompt Engineering [2023-arxiv] [[paper](https://lsiddiqsunny.github.io/public/2304.07840.pdf)] [[repo](https://zenodo.org/records/8122636)]
32. Training Language Models for Programming Feedback Using Automated Repair Tools [2023-AIED] [[paper](https://research.aalto.fi/files/130373931/Training_Language_Models_for_Programming_Feedback_Using_Automated_Repair_Tools.pdf)] [[repo](https://github.com/KoutchemeCharles/aied2023)]
33. RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.15698)] [[repo](https://anonymous.4open.science/r/repairllama-BC13)]
34. Automated Code Editing with Search-Generate-Modify [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.06490)] [[repo](https://github.com/SarGAMTEAM/SarGAM.git)]
35. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.06770)] [[repo](https://www.swebench.com/)]
36. A critical review of large language model on software engineering: An example from chatgpt and automated program repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.08879)] [[repo](https://github.com/iSEngLab/EvalGPTFix)]
37. Neural Program Repair with Program Dependence Analysis and Effective Filter Mechanism [2023-arxiv] [[paper](https://arxiv.org/pdf/2305.09315)]
38. Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.07215)] [[repo](https://github.com/Lune-Blue/COFFEE)]
39. A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair [2023-arxiv]  [[paper](https://arxiv.org/pdf/2304.08191)]
40. STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.14460)]
41. RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.17077)]
42. An Extensive Study on Model Architecture and Program Representation in the Domain of Learning-based Automated Program Repair [2023-APR] [[paper](https://ieeexplore.ieee.org/abstract/document/10189328)] [[repo](https://github.com/AAI-USZ/APR23-representations)]
43. FixEval: Execution-based Evaluation of Program Fixes for Programming Problems [2023-APR] [[paper](https://arxiv.org/pdf/2206.07796)] [[repo](https://github.com/mahimanzum/FixEval)]
44. Evaluating Pre-trained Language Models for Repairing API Misuses [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.16390)] [[repo](https://anonymous.4open.science/r/TOSEM-API-Misuse)]
45. The potential use of ChatGPT for debugging and bug fixing [2023-] [[paper](https://oulurepo.oulu.fi/bitstream/handle/10024/44572/nbnfi-fe20231006139025.pdf?sequence=1)]
46. An Empirical Study of Adoption of ChatGPT for Bug Fixing among Professional Developers [2023-ITA] [[paper](https://bergersci.com/index.php/jta/article/download/19/20)]
47. SelfEvolve: A Code Evolution Framework via Large Language Models [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.02907)]
48. AgentCoder: Multi Agent-Code Generation with Iterative Testing and Optimisation [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.13010)]
49. RTLFixer: Automatically Fixing RTL Syntax Errors with Large Language Models [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.16543)]
50. Can large language models find and fix vulnerable software? [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.10345)]
51. LLM4SecHW: Leveraging domain-specific large language model for hardware debugging [2023-AsianHOST] [[paper](https://arxiv.org/pdf/2401.16448)]
52. Can LLMs Patch Security Issues [2023-arxiv] [[paper](https://arxiv.org/html/2312.00024v2)] [[repo](https://github.com/Kamel773/LLM-code-refine)]
53. Is Self-Repair a Silver Bullet for Code Generation? [2023-ICLR] [[paper](https://openreview.net/pdf?id=y0GJXRungR)] [[repo](https://github.com/theoxo/self-repair)]
54. PatchZero: Zero-Shot Automatic Patch Correctness Assessment [2023-arxiv] [[paper](https://arxiv.org/pdf/2303.00202)]
55. A Novel Approach for Automatic Program Repair using Round-Trip Translation with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.07994)] [[repo](https://zenodo.org/records/10500594)]
56. Enhanced Automated Code Vulnerability Repair using Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.03741)]
57. DebugBench: Evaluating Debugging Capability of Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.04621)] [[repo](https://github.com/thunlp/DebugBench)]
58. Automated Test Case Repair Using Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.06765)]
59. ContrastRepair: Enhancing Conversation-Based Automated Program Repair via Contrastive Test Case Pairs [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.01971)]
60. Exploring the Potential of Pre-Trained Language Models of Code for Automated Program Repair [2024-Electronics] [[paper](https://www.mdpi.com/2079-9292/13/7/1200)]
61. CigaR: Cost-efficient Program Repair with LLMs [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.06598)] [[repo](https://github.com/ASSERT-KTH/cigar)]
62. A Novel Approach for Automated Program Repair using Round-Trip Translation with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.07994)] [[repo](https://zenodo.org/records/10500594)]
63. ConDefects: A New Dataset to Address the Data Leakage Concern for LLM-based Fault Localization and Program Repair [2024-arxiv] [[paper](https://arxiv.org/pdf/2310.16253)] [[repo](https://github.com/appmlk/ConDefects)]
64. HDLdebugger: Streamlining HDL debugging with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.11671)]
65. DrPlanner: Diagnosis and Repair of Motion Planners Using Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.07470)] [[repo](https://github.com/CommonRoad/drplanner)]
66. GitBug-Java: A Reproducible Benchmark of Recent Java Bugs [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.02961)] [[repo](https://github.com/gitbugactions/gitbug-java)]
67. RepairAgent: An Autonomous, LLM-Based Agent for Program Repair [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.17134)]
68. Peer-aided Repairer: Empowering Large Language Models to Repair Advanced Student Assignments [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.01754)]
69. Improved Program Repair Methods using Refactoring with GPT Models [2024-SIGCSE TS] [[paper](https://dl.acm.org/doi/pdf/10.1145/3626252.3630875)] [[repo](https://github.com/RYOSKATE/refactory-with-gpt)]
70. A Study of Vulnerability Repair in JavaScript Programs with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.13193)] [[repo](https://doi.org/10.5281/zenodo.10783763)]
71. ACFIX: Guiding LLMs with Mined Common RBAC Practices for Context-Aware Repair of Access Control Vulnerabilities in Smart Contracts [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.06838)]
72. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? [2024-ICLR] [[paper](https://arxiv.org/pdf/2310.06770)] [[repo](https://github.com/princeton-nlp/SWE-bench)]
73. CodeEditorBench: Evaluating Code Editing Capability of Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.03543)] [[repo](https://github.com/CodeEditorBench/CodeEditorBench)]
74. Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.17230)]
75. Teaching Large Language Models to Self-Debug [2024-ICLR] [[paper](https://arxiv.org/pdf/2304.05128)]
76. OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.14658)] [[repo](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter)]
77. LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.16906)] [[repo](https://github.com/FloridSleeves/LLMDebugger)]
78. Leveraging Print Debugging to Improve Code Generation in Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.05319)]

