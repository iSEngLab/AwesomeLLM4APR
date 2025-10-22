<h1 align = "center">🤖 Awesome LLM for APR</h1>
<p align="center">
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"></a>
  <a href="https://arxiv.org/abs/2405.01466"><img src="https://img.shields.io/badge/arXiv-2405.01466-blue.svg"></a>
  <img src="https://img.shields.io/github/stars/iSEngLab/AwesomeLLM4APR?color=yellow&label=Stars">
  <img src="https://img.shields.io/badge/PRs-Welcome-red">
  <img src="https://img.shields.io/github/last-commit/iSEngLab/AwesomeLLM4APR">
</p>



**We use an LLM-based bot to automatically fetch and summarize new LLM4APR papers, with regular human curation to ensure quality. You can check the raw bot updates in this separate [update_file](https://github.com/iSEngLab/AwesomeLLM4APR/blob/main/update.md), or explore the curated summaries on our [summary site](https://iseabot.github.io/CI-LLM4APR).**

## 📖 Contents

- [👏 Citation](#-citation)
- [💡 Repair Scenarios](#-repair-scenarios)
  - [Semantic Bug](#semantic-bug)
  - [Security Vulnerability](#security-vulnerability)
  - [Syntax Error](#syntax-error)
  - [Programming Problem](#programming-problem)
  - [Static Warning](#static-warning)
  - [Self-Debug](#self-debug)
  - [Type Error](#type-error)
  - [Web UI Test](#web-ui-test)
  - [Repository-level Issue](#repository-level-issue)
  - [Smart Contract](#smart-contract)
  - [Hardware Bug](#hardware-bug)
  - [Performance Bug](#performance-bug)
  - [API Misuse](#api-misuse)
  - [Crash Bug](#crash-bug)
  - [Test Case](#test-case)
  - [Error-handling Bug](#error-handling-bug)
  - [Formal Proof](#formal-proof)
  - [Translation Bug](#translation-bug)
  - [GitHub Issue](#github-issue)
  - [Code Review](#code-review)
  - [Motion Planner](#motion-planner)
- [🙆 Human Study](#-human-study)
- [🙅 Patch Correctness Assessment](#-patch-correctness-assessment)
- [📊 Benchmark](#-benchmark)
- [🤔 Related APR Surveys](#-related-apr-surveys)


## 👏 Citation

```bibtex
@article{zhang2024survey,
  title={A Systematic Literature Review on Large Language Models for Automated Program Repair},
  author={Zhang, Quanjun and Fang, Chunrong and Xie, Yang and Ma, Yuxiang and Sun, Weisong and Yang, Yun and Chen, Zhenyu},
  journal={arXiv preprint arXiv:2405.01466}
  year={2024}
}

```

# 🚗Todo List

- [ ] add SE agent-based studies for GitHub Issues
- [ ] add ISSTA 2024 Papers
- [ ] release a new version of this paper on arXiv

## 🔥🔥 New Papers

1. 🔥A Comprehensive Survey of AI-Driven Advancements and Techniques in Automated Program Repair and Code Generation  [2024-arXiv]
2. 🔥SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents [2024-NeurIPS]
3. 🔥CORE: Resolving Code Quality Issues using LLMs [2024-FSE]
4. 🔥Prompt Fix: Vulnerability Automatic Repair Technology Based on Prompt Engineering [2024-ICNC]
5. 🔥Evaluating Large Language Models for Real-World Vulnerability Repair in C/C++ Code[2024-IWSPA]
6. 🔥Investigating large language models capabilities for automatic code repair in Python[2024-Cluster Computing]
7. 🔥LPR: Large Language Models-Aided Program Reduction[2024-ISSTA]
8. 🔥A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback (2024年7月) [AIware 2024](https://dl.acm.org/doi/proceedings/10.1145/3664646)
9. 🔥When Large Language Models Confront Repository-Level Automatic Program Repair: How Well They Done? [2024-ICSE]
10. 🔥Exploring Parameter-Efficient Fine-Tuning of Large Language Model on Automated Program Repair[2024-ASE]
11. Exploring the Potential of Conversational Test Suite Based Program Repair on SWE-bench [2024-arXiv]
12. Exploring and Lifting the Robustness of LLM-powered Automated Program Repair with Metamorphic Testing[2024-arXiv] [[paper](https://arxiv.org/abs/2410.07516)]
13. Divide-and-Conquer: Automating Code Revisions via Localization-and-Revision [2024-TOSEM]
14. From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging [2024-arXiv] [[paper](https://arxiv.org/abs/2410.01215)] [[repo](https://github.com/YerbaPage/MGDebugger)]
15. Automated Program Repair for Introductory Programming Assignments [2024-TLT] [[paper](https://ieeexplore.ieee.org/document/10535720/)]
16. Automated Repair of AI Code with Large Language Models and Formal Verification [2024-arXiv] [[paper](https://arxiv.org/abs/2405.08848)]
17. CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair [2024-arXiv-NVIDIA] [[paper](https://arxiv.org/abs/2409.12993)]
18. Benchmarking Automated Program Repair: An Extensive Study on Both Real-World and Artificial Bugs [2024-ISSTA]  [[paper](https://dl.acm.org/doi/10.1145/3650212.3652140)]
19. Automated program repair via conversation: Fixing 162 out of 337 bugs for $0.42 each using chatgpt[2024-ISSTA] [[paper](https://dl.acm.org/doi/10.1145/3650212.3680323)]
20. Leveraging Large Language Model for Automatic Patch Correctness Assessment[2024-TSE] [[paper](https://ieeexplore.ieee.org/document/10659742)]
21. Automated program repair for variability bugs in software product line systems[2024-JSS] [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0164121224001973)]
22. PyBugHive: A Comprehensive Database of Manually Validated, Reproducible Python Bugs[2024-IEEE Access] [[paper](https://ieeexplore.ieee.org/document/10644000)]
23. How to Understand Whole Software Repository? [2024-arXiv] [[paper](https://arxiv.org/pdf/2406.01422)]

## 💡 Repair Scenarios 

### Semantic Bug

1. From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging [2024-arxiv] [[repo](https://github.com/YerbaPage/MGDebugger)]
2. When Large Language Models Confront Repository-Level Automatic Program Repair: How Well They Done? [2024-ICSE] [[repo](N.A.)]
3. T5APR: Empowering Automated Program Repair across Languages through Checkpoint Ensemble [2024-JSS] [[repo](https://github.com/h4iku/T5APR)]
4. A Deep Dive into Large Language Models for Automated Bug Localization and Repair [2024-FSE/ESEC] [[repo](N.A.)]
5. Benchmarking Automated Program Repair: An Extensive Study on Both Real-World and Artificial Bugs [2024-ISSTA] [[repo](N.A)]
6. Automated Program Repair via Conversation: Fixing 162 out of 337 bugs for $0.42 each using chatgpt [2024-ISSTA] [[repo](N.A)]
7. How Far Can We Go with Practical Function-Level Program Repair? [2024-arxiv] [[repo](https://github.com/GhabiX/SRepair)]
8. Exploring and Lifting the Robustness of LLM-powered Automated Program Repair with Metamorphic Testing [2024-arxiv] [[repo](N.A.)]
9. Thinkrepair: Self-directed automated program repair [2024-ISSTA] [[repo](https://github.com/vinci-grape/ThinkRepair)]
10. Hierarchical Knowledge Injection for Improving LLM-based Program Repair [2025-ASE] [[repo](https://github.com/SOAR-Lab/llm-apr-knowledge-injection)]
11. Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair [2025-TOSEM] [[repo](https://github.com/XYZboom/DEVLoRe)]
12. APRMCTS: Improving LLM-based Automated Program Repair with Iterative Tree Search [2025-ASE] [[repo](https://github.com/Tomsawyerhu/APR-MCTS)]
13. Demystifying Memorization in LLM-based Program Repair via a General Hypothesis Testing Framework [2025-FSE/ESEC] [[repo](https://sites.google.com/view/memprompt)]
14. Hybrid Automated Program Repair by Combining Large Language Models and Program Analy [2025-TOSEM] [[repo](https://github.com/Feng-Jay/GiantRepair)]
15. When Fine-Tuning LLMs Meets Data Privacy: An Empirical Study of Federated Learning in LLM-Based Program Repair [2025-TOSEM] [[repo](https://github.com/stringing/Federated-LLM-Based-APR)]
16. The Impact of Fine-tuning Large Language Models on Automated Program Repair [2025-ICSME] [[repo](https://doi.org/10.5281/zenodo.16359186)]
17. Knowledge-Enhanced Program Repair for Data Science Code [2025-ICSE] [[repo](https://github.com/ShuyinOuyang/DSrepair)]
18. The Fact Selection Problem in LLM-Based Program Repair [2025-ICSE] [[repo](https://github.com/PyRepair/maniple)]
19. The Art of Repair: Optimizing Iterative Program Repair with Instruction-Tuned Models [2025-EASE] [[repo](https://doi.org/10.5281/zenodo.15294695)]
20. MORepair: Teaching LLMs to Repair Code via Multi-Objective Fine-tuning [2025-TOSEM] [[repo](N.A.)]
21. Adversarial Reasoning for Repair Based on Inferred Program Intent [2025-ISSTA] [[repo](https://doi.org/10.5281/zenodo.15367930)]
22. Repair Ingredients Are All You Need: Improving Large Language Model-Based Program Repair viaRepair Ingredients Search [2025-ICSE] [[repo](https://sites.google.com/view/repairingredients)]
23. Aligning the Objective of LLM-based Program Repair [2025-ICSE] [[repo](https://github.com/CUHK-Shenzhen-SE/D4C)]
24. One Size Does Not Fit All: Multi-granularity Patch Generation for Better Automated Program Repair [2024-ISSTA] [[repo](https://zenodo.org/records/12660892)]
25. Template-Guided Program Repair in the Era of Large Language Models [2025-ICSE] [[repo](https://sites.google.com/view/neuraltemplaterepair)]
26. Revisiting Unnaturalness for Automated Program Repair in the Era of Large Language Models [2024-arxiv] [[repo](https://zenodo.org/records/10851256)]
27. HapRepair: Learn to Repair OpenHarmony Apps [2025-FSE/ESEC] [[repo](https://github.com/SMAT-Lab/HapRepair)]
28. Automated program repair for variability bugs in software product line systems[2024-JSS] [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0164121224001973)]
29. A Unified Debugging Approach via LLM-Based Multi-Agent Synergy [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.17153)] [[repo](https://github.com/afortunado-aceptado/Rudra)]
30. How Far Can We Go with Practical Function-Level Program Repair? [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.12833)] [[repo](https://github.com/GhabiX/SRepair)]
31. Automated program repair via conversation: Fixing 162 out of 337 bugs for $0.42 each using chatgpt[2024-ISSTA] [[paper](https://dl.acm.org/doi/10.1145/3650212.3680323)]
   <br> Old Version: Keep the Conversation Going: Fixing 162 out of 337 bugs for $0.42 each using ChatGPT [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.00385)]
32. A Novel Approach for Automatic Program Repair using Round-Trip Translation with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.07994)] [[repo](https://zenodo.org/records/10500594)]
33. Out of Context: How important is Local Context in Neural Program Repair? [2024-ICSE] [[paper](https://arxiv.org/pdf/2312.04986)] [[repo](https://github.com/giganticode/out_of_context_paper_data)]
34. Multi-Objective Fine-Tuning for Enhanced Program Repair with LLMs [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.12636)]
35. Aligning the Objective of LLM-based Program Repair [2025-ICSE] [[paper](https://arxiv.org/pdf/2404.08877)] [[repo](https://github.com/CUHK-Shenzhen-SE/D4C)]
36. ContrastRepair: Enhancing Conversation-Based Automated Program Repair via Contrastive Test Case Pairs [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.01971)]
37. Exploring the Potential of Pre-Trained Language Models of Code for Automated Program Repair [2024-Electronics] [[paper](https://www.mdpi.com/2079-9292/13/7/1200)]
38. CigaR: Cost-efficient Program Repair with LLMs [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.06598)] [[repo](https://github.com/ASSERT-KTH/cigar)]
39. The Fact Selection Problem in LLM-Based Program Repair [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.05520)] [[repo](https://github.com/PyRepair/maniple)]
40. A Novel Approach for Automated Program Repair using Round-Trip Translation with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.07994)] [[repo](https://zenodo.org/records/10500594)]
41. RepairAgent: An Autonomous, LLM-Based Agent for Program Repair [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.17134)]
42. A Deep Dive into Large Language Models for Automated Bug Localization and Repair [2024-FSE/ESEC] [[paper](https://arxiv.org/pdf/2404.11595)]
43. Automated Program Repair in the Era of Large Pre-trained Language Models [2023-ICSE] [[paper](https://web.eecs.umich.edu/~movaghar/IEEE-2023-Automated_Program_Repair_in_the_Era_of_Large_Pre-trained_Language_Models.pdf)] [[repo](https://zenodo.org/records/7592886)]
44. Repair Is Nearly Generation: Multilingual Program Repair with LLMs [2023-AAAI] [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/25642/25414)]
45. Retrieval-based prompt selection for code-related few-shot learning [2023-ICSE] [[paper](https://nashid.github.io/resources/papers/cedar-icse23.pdf)] [[repo](https://github.com/prompt-learning/cedar)]
46. What makes good in-context demonstrations for code intelligence tasks with llms? [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298329)] [[repo](https://github.com/shuzhenggao/ICL4code)]
47. Fully Autonomous Programming with Large Language Models [2023-GECCO] [[paper](https://dl.acm.org/doi/pdf/10.1145/3583131.3590481)] [[repo](https://github.com/KoutchemeCharles/aied2023)]
48. Automated Program Repair Using Generative Models for Code Infilling [2023-AIED] [[paper](https://link.springer.com/chapter/10.1007/978-3-031-36272-9_74)] [[repo](https://github.com/KoutchemeCharles/aied2023)]
49. STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.14460)]
50. Conversational automated program repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2301.13246)]
51. Is ChatGPT the Ultimate Programming Assistant--How far is it? [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.11938)] [[repo](https://github.com/HaoyeTianCoder/ChatGPT-Study)]
52. Using Large Language Models for Bug Localization and Fixing [2023-iCAST] [[paper](https://u-aizu.ac.jp/~markov/pubs/iCAST_23.pdf)]
53. An Empirical Study on Fine-Tuning Large Language Models of Code for Automated Program Repair [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298532)] [[repo](https://github.com/LLMC-APR/STUDY)]
54. An Evaluation of the Effectiveness of OpenAI's ChatGPT for Automated Python Program Bug Fixing using QuixBugs [2023-iSEMANTIC] [[paper](https://ieeexplore.ieee.org/abstract/document/10295323)]
55. Explainable Automated Debugging via Large Language Model-driven Scientific Debugging [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.02195)]
56. The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.17485)]
57. Impact of Code Language Models on Automated Program Repair [2023-ICSE] [[paper](https://arxiv.org/pdf/2302.05020)] [[repo](https://github.com/lin-tan/clm)]
58. Towards Generating Functionally Correct Code Edits from Natural Language Issue Descriptions [2023-arxiv] [[paper](https://arxiv.org/pdf/2304.03816)]
59. The Plastic Surgery Hypothesis in the Era of Large Language Models [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298499)] [[repo](https://zenodo.org/records/8244813)]
60. Exploring the Limits of ChatGPT in Software Security Applications [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.05275)]
61. CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.08588)] [[repo](https://github.com/WeixiangYAN/CodeScope)]
62. Enhancing Automated Program Repair through Fine-tuning and Prompt Engineering [2023-arxiv] [[paper](https://lsiddiqsunny.github.io/public/2304.07840.pdf)] [[repo](https://zenodo.org/records/8122636)]
63. Training Language Models for Programming Feedback Using Automated Repair Tools [2023-AIED] [[paper](https://research.aalto.fi/files/130373931/Training_Language_Models_for_Programming_Feedback_Using_Automated_Repair_Tools.pdf)] [[repo](https://github.com/KoutchemeCharles/aied2023)]
64. RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.15698)] [[repo](https://anonymous.4open.science/r/repairllama-BC13)]
65. Automated Code Editing with Search-Generate-Modify [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.06490)] [[repo](https://github.com/SarGAMTEAM/SarGAM.git)]
66. RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.06057)] [[repo](https://figshare.com/s/a4e95baee01bba14bf4b)]
67. Neural Program Repair with Program Dependence Analysis and Effective Filter Mechanism [2023-arxiv] [[paper](https://arxiv.org/pdf/2305.09315)]
68. Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.07215)] [[repo](https://github.com/Lune-Blue/COFFEE)]
69. A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair [2023-arxiv]  [[paper](https://arxiv.org/pdf/2304.08191)]
70. Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.00608)] [[repo](https://github.com/ise-uiuc/Repilot)]
71. Gamma: Revisiting Template-Based Automated Program Repair Via Mask Prediction [2023-ASE] [[paper](https://arxiv.org/pdf/2309.09308)] [[repo](https://github.com/iSEngLab/GAMMA)]
72. An Extensive Study on Model Architecture and Program Representation in the Domain of Learning-based Automated Program Repair [2023-APR] [[paper](https://ieeexplore.ieee.org/abstract/document/10189328)] [[repo](https://github.com/AAI-USZ/APR23-representations)]
73. Improving Automated Program Repair with Domain Adaptation [2023-TOSEM] [[paper](https://arxiv.org/pdf/2212.11414)] [[repo](https://github.com/arminzirak/TFix)]
74. Enhancing Code Language Models for Program Repair by Curricular Fine-tuning Framework [2023-ICSME] [[paper](https://ieeexplore.ieee.org/abstract/document/10336339)]
75. The potential use of ChatGPT for debugging and bug fixing [2023-] [[paper](https://oulurepo.oulu.fi/bitstream/handle/10024/44572/nbnfi-fe20231006139025.pdf?sequence=1)]
76. CIRCLE: Continual Repair across Programming Languages [2022-ISSTA] [[paper](https://arxiv.org/pdf/2205.10956)] [[repo](https://github.com/2022CIRCLE/CIRCLE)]
77. Towards JavaScript program repair with Generative Pre-trained Transformer (GPT-2) [2022-APR] [[paper](http://publicatio.bibl.u-szeged.hu/25241/1/Lajko-APR22.pdf)] [[repo](https://github.com/AAI-USZ/APR22-JS-GPT)]
78. Fix Bugs with Transformer through a Neural-Symbolic Edit Grammar [2022-ICLR] [[paper](https://arxiv.org/pdf/2204.06643)]
79. Patch Generation with Language Models: Feasibility and Scaling Behavior [2022-ICLR] [[paper](https://openreview.net/pdf?id=rHlzJh_b1-5)]
80. Can OpenAI's codex fix bugs?: an evaluation on QuixBugs [2022-APR] [[paper](https://dl.acm.org/doi/abs/10.1145/3524459.3527351)]
81. An Analysis of the Automatic Bug Fixing Performance of ChatGPT [2022-APR] [[paper](https://ieeexplore.ieee.org/abstract/document/10189263)] [[repo](https://gitlab.rlp.net/dsobania/chatgpt-apr)]
82. Less training, more repairing please: revisiting automated program repair via zero-shot learning [2022-FSE/ESEC] [[paer](https://dl.acm.org/doi/pdf/10.1145/3540250.3549101)] [[repo](https://zenodo.org/records/6819444)]
83. Framing Program Repair as Code Completion [2022-APR] [[paper](http://repositorio.inesctec.pt/bitstreams/2fb3b152-a3ba-4561-ad11-3869b0d245a0/download)] [[repo](https://github.com/FranciscoRibeiro/code-truncater)]
84. DEAR A Novel Deep Learning-based Approach for Automated Program Repair [2022-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3510003.3510177)] [[repo](https://github.com/AutomatedProgramRepair-2021/dear-auto-fix)]
85. Generating Bug-Fixes Using Pretrained Transformers [2021-PLDI] [[paper](https://arxiv.org/pdf/2104.07896)]
86. Applying CodeBERT for Automated Program Repair of Java Simple Bugs [2021-MSR] [[paper](https://arxiv.org/pdf/2103.11626)] [[repo](https://github.com/EhsanMashhadi/MSR2021-ProgramRepair)]
87. CURE Code-Aware Neural Machine Translation for Automatic Program Repair [2021-ICSE] [[paper](https://arxiv.org/pdf/2103.00073)] [[repo](https://github.com/lin-tan/CURE)]
88. How to Understand Whole Software Repository? [2024-arXiv] [[paper](https://arxiv.org/pdf/2406.01422)]

### Security Vulnerability

1. A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback [2024-AIware] [[repo](https://drive.google.com/drive/folders/1yrYUJS1r2cu7G6D3ezASsfugrNqgwc3v)]
2. VulAdvisor: Natural Language Suggestion Generation for Software Vulnerability Repair [2024-ASE] [[repo](https://github.com/zhangj111/VulAdvisor)]
3. Teaching AI the ‘Why’ and ‘How’ of Software Vulnerability Fixes [2025-FSE/ESEC] [[repo](https://github.com/amiaog/Teaching-AI-the-Why-and-How-of-Software-Vulnerability-Fixes)]
4. APPATCH: Automated Adaptive Prompting Large Language Models for Real-World Software Vulnerability Patching [2025-USENIX Security] [[repo](https://zenodo.org/records/14741018)]
5. Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection & Repair in the IDESecurityArtifact-FunctionalArtifact-AvailableArtifact
6. Reusable [2025-ICSE] [[repo](https://doi.org/10.6084/m9.figshare.26367139)]
7. PATCHAGENT: A Practical Program Repair Agent Mimicking Human Expertise [2025-USENIX Security] [[repo](https://osf.io/8k2ac)]
8. 🔥Automated Repair of AI Code with Large Language Models and Formal Verification [2024-arXiv] [[paper](https://arxiv.org/abs/2405.08848)]
9. 🔥NAVRepair: Node-type Aware C/C++ Code Vulnerability Repair [2024-arxiv] [[paper](https://arxiv.org/abs/2405.04994)]
10. Enhanced Automated Code Vulnerability Repair using Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.03741)]
11. Out of Sight, Out of Mind: Better Automatic Vulnerability Repair by Broadening Input Ranges and Sources [2024-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3639222)] [[repo](https://github.com/soarsmu/VulMaster_)]
12. A Study of Vulnerability Repair in JavaScript Programs with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.13193)] [[repo](https://doi.org/10.5281/zenodo.10783763)]
13. Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.17230)]
14. Pre-trained Model-based Automated Software Vulnerability Repair: How Far are We? [2023-TDSC] [[paper](https://arxiv.org/pdf/2308.12533)] [[repo](https://github.com/iSEngLab/LLM4VulFix)]
15. Examining zero-shot vulnerability repair with large language models [2023-S&P] [[paper](https://arxiv.org/pdf/2112.02125)] [[repo](https://drive.google.com/drive/folders/1xJ-z2Wvvg7JSaxfTQdxayXFEmoF3y0ET?usp=sharing)]
16. An Empirical Study on Fine-Tuning Large Language Models of Code for Automated Program Repair [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298532)] [[repo](https://github.com/LLMC-APR/STUDY)]
17. A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification [2023-arxiv] [[paper](https://arxiv.org/pdf/2305.14752)]
18. Exploring the Limits of ChatGPT in Software Security Applications [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.05275)]
19. ZeroLeak: Using LLMs for Scalable and Cost Effective Side-Channel Patching [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.13062)]
20. How ChatGPT is Solving Vulnerability Management Problem [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.06530)] [[repo](https://anonymous.4open.science/r/DefectManagementEvaluation-0411)]
21. How Effective Are Neural Networks for Fixing Security Vulnerabilities [2023-ISSTA] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597926.3598135)] [[repo](https://github.com/lin-tan/llm-vul)]
22. Vision Transformer-Inspired Automated Vulnerability Repair [2023-TOSEM] [[paper](https://www.researchgate.net/profile/Michael-Fu-8/publication/375618720_Vision_Transformer-Inspired_Automated_Vulnerability_Repair/links/65a9b875ee1e1951fbbe6538/Vision-Transformer-Inspired-Automated-Vulnerability-Repair.pdf)] [[repo](https://github.com/awsm-research/VQM)]
23. Can large language models find and fix vulnerable software? [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.10345)]
24. VulRepair: A T5-Based Automated Software Vulnerability Repair [2022-FSE/ESEC] [[paper](https://www.researchgate.net/profile/Chakkrit-Tantithamthavorn/publication/362092639_VulRepair_A_T5-Based_Automated_Software_Vulnerability_Repair/links/6345ea1076e39959d6b73228/VulRepair-A-T5-Based-Automated-Software-Vulnerability-Repair.pdf)] [[repo](https://github.com/awsm-research/VulRepair)]

    

### Syntax Error

1. A Novel Approach for Automated Program Repair using Round-Trip Translation with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.07994)] [[repo](https://zenodo.org/records/10500594)]
2. Repair Is Nearly Generation: Multilingual Program Repair with LLMs [2023-AAAI] [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/25642/25414)]
3. Fixing Rust Compilation Errors using LLMs [2023-arxiv] [[paper](https://arxiv.org/pdf/2308.05177)]
4. An Empirical Study on Fine-Tuning Large Language Models of Code for Automated Program Repair [2023-ASE] [[paper](https://ieeexplore.ieee.org/abstract/document/10298532)] [[repo](https://github.com/LLMC-APR/STUDY)]
5. A Chain of AI-based Solutions for Resolving FQNs and Fixing Syntax Errors in Partial Code [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.11981)] [[repo](https://github.com/SE-qinghuang/A-Chain-of-AI-based-Solutions-for-Resolving-FQNs-and-Fixing-Syntax-Errors-in-Partial-Code)]
6. The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.17485)]
7. SYNSHINE: improved fixing of Syntax Errors [2022-TSE] [[paper](https://ieeexplore.ieee.org/abstract/document/9913705)] [[repo](https://zenodo.org/record/4572390#.Y4CY8xRByUk)]

### Programming Problem

1. 🔥Combining Logic and Large Language Models for Assisted Debugging and Repair of ASP Programs [2025-ICST] [[repo](https://github.com/RicardoBrancas/formhe)]
2. 🔥Less is More: Adaptive Program Repair with Bug Localization and Preference Learning [2025-AAAI] [[repo](https://github.com/zhenlongDai/)]
3. 🔥Exploring Parameter-Efficient Fine-Tuning of Large Language Model on Automated Program Repair [2024-ASE] [[repo](https://github.com/zjulgc/llmpeft4apr)]
4. 🔥FastFixer: An Efficient and Effective Approach for Repairing Programming Assignments [2024-ASE] [[repo](https://github.com/LiuFang816/FastFixer)]
5. 🔥Investigating Large Language Models Capabilities for Automatic Code Repair in Python [2024-Cluster Computing] [[repo](https://github.com/KshitizBasnet2021/ChatGPTResearch)]
6. 🔥Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization [2025-AAAI] [[repo](https://github.com/pmorvalho/LLM-CEGIS-Repair)]
7. 🔥Code repair with llms gives an exploration-exploitation tradeoff [2024-NeurIPS] [[repo](https://github.com/haotang1995/REx)]
8. 🔥Automated Program Repair for Introductory Programming Assignments [2024-TLT] [[repo](N.A)]
9. 🔥Investigating the Transferability of Code Repair for Low-Resource Programming Languages [2025-NAACL] [[repo](https://github.com/KyleWong288/Distill_LRPL)]
10. 🔥CREF: An LLM-based Conversational Software Repair Framework for Programming Tutors [2024-ISSTA] [[repo](https://github.com/buaabarty/CREF)]
11. 🔥RePair: Automated Program Repair with Process-based Feedback [2024-ACL] [[repo](https://github.com/TnTWoW/RePair)]
12. CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair [2024-arXiv-NVIDIA] [[paper](https://arxiv.org/abs/2409.12993)]
13. A Unified Debugging Approach via LLM-Based Multi-Agent Synergy [2024-arXiv] [[paper](https://arxiv.org/pdf/2404.17153)] [[repo](https://github.com/afortunado-aceptado/Rudra)]
14. PyDex: Repairing Bugs in Introductory Python Assignments using LLMs [2024-OOPSLA] [[paper](https://dl.acm.org/doi/pdf/10.1145/3649850)] [[repo](https://github.com/microsoft/prose-benchmarks/tree/main/PyDex)]
15. DebugBench: Evaluating Debugging Capability of Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.04621)] [[repo](https://github.com/thunlp/DebugBench)]
16. ContrastRepair: Enhancing Conversation-Based Automated Program Repair via Contrastive Test Case Pairs [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.01971)]
17. ConDefects: A New Dataset to Address the Data Leakage Concern for LLM-based Fault Localization and Program Repair [2024-arxiv] [[paper](https://arxiv.org/pdf/2310.16253)] [[repo](https://github.com/appmlk/ConDefects)]
18. Peer-aided Repairer: Empowering Large Language Models to Repair Advanced Student Assignments [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.01754)]
19. Improved Program Repair Methods using Refactoring with GPT Models [2024-SIGCSE TS] [[paper](https://dl.acm.org/doi/pdf/10.1145/3626252.3630875)] [[repo](https://github.com/RYOSKATE/refactory-with-gpt)]
20. A critical review of large language model on software engineering: An example from chatgpt and automated program repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.08879)] [[repo](https://github.com/iSEngLab/EvalGPTFix)]
21. Automated Repair of Programs from Large Language Models [2023-ICSE] [[paper](https://arxiv.org/pdf/2205.10583)] [[repo](https://github.com/zhiyufan/apr4codex)]
22. FixEval: Execution-based Evaluation of Program Fixes for Programming Problems [2023-APR] [[paper](https://arxiv.org/pdf/2206.07796)] [[repo](https://github.com/mahimanzum/FixEval)]
23. Refining ChatGPT-Generated Code: Characterizing and Mitigating Code Quality Issues [2023-TOSEM] [[paper](https://dl.acm.org/doi/pdf/10.1145/3643674)] [[repo](https://github.com/yueyueL/ChatGPT-CodeGenAnalysis)]
24. Repairing bugs in python assignments using large language models [2022-arixv] [[paper](https://arxiv.org/pdf/2209.14876)]

### Static Warning

1. CORE: Resolving Code Quality Issues using LLMs [2024-FSE/ESEC]
2. Frustrated with Code Quality Issues? LLMs can Help! [2024-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.12938)] [[repo](https://aka.ms/CORE_MSRI)]
3. SkipAnalyzer: An Embodied Agent for Code Analysis with Large Language Models [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.18532)] [[repo](https://zenodo.org/records/10043170)]
4. RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2309.06057)] [[repo](https://figshare.com/s/a4e95baee01bba14bf4b)]
5. InferFix: End-to-End Program Repair with LLMs over Retrieval-Augmented Prompts [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2303.07263)] [[repo](https://github.com/microsoft/InferredBugs)]
6. Can LLMs Patch Security Issues [2023-arxiv] [[paper](https://arxiv.org/html/2312.00024v2)] [[repo](https://github.com/Kamel773/LLM-code-refine)]
7. Improving Automated Program Repair with Domain Adaptation [2023-TOSEM] [[paper](https://arxiv.org/pdf/2212.11414)] [[repo](https://github.com/arminzirak/TFix)]
8. An empirical study of deep transfer learning-based program repair for Kotlin projects [2022-FSE/ESEC] [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3558967)]
9. TFix-Learning to Fix Coding Errors with a Text-to-Text Transformer [2021-PMLR] [[paper](http://proceedings.mlr.press/v139/berabi21a/berabi21a.pdf)] [[repo](https://github.com/eth-sri/TFix)]

### Self-Debug

1. From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging [2024-arxiv] [[repo](https://github.com/YerbaPage/MGDebugger)]
2. CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair [2024-ICLR]
3. From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging [2024-arXiv] [[paper](https://arxiv.org/abs/2410.01215)] [[repo](https://github.com/YerbaPage/MGDebugger)]
4. Teaching Large Language Models to Self-Debug [2024-ICLR] [[paper](https://arxiv.org/pdf/2304.05128)]
5. OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.14658)] [[repo](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter)]
6. CYCLE: Learning to Self-Refine the Code Generation [2024-OOPSLA] [[paper](https://arxiv.org/pdf/2403.18746)] [[repo](https://github.com/ARiSE-Lab/CYCLE_OOPSLA_24)]
7. LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.16906)] [[repo](https://github.com/FloridSleeves/LLMDebugger)]
8. Leveraging Print Debugging to Improve Code Generation in Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.05319)]
9. SelfEvolve: A Code Evolution Framework via Large Language Models [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.02907)]
10. Self-Refine: Iterative Refinement with Self-Feedback [2023-NeurIPS] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf)] [[repo](https://github.com/madaan/self-refine)]
11. AgentCoder: Multi Agent-Code Generation with Iterative Testing and Optimisation [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.13010)]
12. Self-Edit: Fault-Aware Code Editor for Code Generation [2023-ACL] [[paper](https://arxiv.org/pdf/2305.04087)] [[repo](https://github.com/zkcpku/Self-Edit)]
13. Is Self-Repair a Silver Bullet for Code Generation? [2023-ICLR] [[paper](https://openreview.net/pdf?id=y0GJXRungR)] [[repo](https://github.com/theoxo/self-repair)]


### Type Error

1. RetypeR: Integrated Retrieval-based Automatic Program Repair for Python Type Errors [2024-ICSME] [[repo](https://anonymous.4open.science/r/RetypeR)]
2. Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors [2024-ICSE] [[paper](https://arxiv.org/pdf/2306.01394)] [[repo](https://github.com/JohnnyPeng18/TypeFix)]
3. PyTy: Repairing Static Type Errors in Python [2024-ICSE] [[paper](https://arxiv.org/pdf/2401.06619)] [[repo](https://github.com/sola-st/PyTy)]
4. GPT-3-Powered Type Error Debugging: Investigating the Use of Large Language Models for Code Repair [2023-SLE] [[paper](https://dl.acm.org/doi/abs/10.1145/3623476.3623522)] [[repo](https://gitlab.com/FranciscoRibeiro/mentat)]

### Web UI Test

1. Guiding ChatGPT to Fix Web UI Tests via Explanation-Consistency Checking [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.05778)]

### Repository-level Issue

1. MASAI: Modular Architecture for Software-engineering AI Agents [2024-NeurIPS] [[repo](N.A.)]
2. CodeR: Issue Resolving with Multi-Agent and Task Graphs [2024-arxiv] [[repo](https://github.com/NL2Code/CodeR)]
3. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering [2024-NeurIPS] [[repo](https://github.com/SWE-agent/SWE-agent)]
4. AutoCodeRover: Autonomous Program Improvement [2024-ISSTA] [[repo](https://autocoderover.dev/)]
5. MarsCode Agent: AI-native Automated Bug Fixing [2024-arxiv] [[repo](N.A.)]
6. Enhancing Automated Program Repair with Solution Design [2024-ASE] [[repo](https://figshare.com/s/82ed8e86e88d3268b4c1)]
7. Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution [2025-MSR] [[repo](https://anonymous.4open.science/r/prompt-knowledge-gap-BE45/README.md)]
8. OmniGIRL: A Multilingual and Multimodal Benchmark for GitHub Issue Resolution [2025-ISSTA] [[repo](https://github.com/DeepSoftwareAnalytics/OmniGIRL)]
9. SWE-GPT: A Process-Centric Language Model for Automated Software Improvement [2025-ISSTA] [[repo](https://github.com/LingmaTongyi/Lingma-SWE-GPT)]
10. SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents [2025-NeurIPS] [[repo](https://github.com/logic-star-ai/SWT-Bench)]
11. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement [2025-ICLR] [[repo](https://github.com/aorwall/moatless-tree-search)]
12. RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph [2025-ICLR] [[repo](https://github.com/ozyyshr/RepoGraph)]
13. SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution [2025-NeurIPS] [[repo](https://github.com/facebookresearch/swe-rl)]
14. Demystifying LLM-based Software Engineering Agents [2025-FSE/ESEC] [[repo](https://github.com/OpenAutoCoder/Agentless)]
15. MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution [2025-NeurIPS] [[repo](N.A)]
16. OpenHands: An Open Platform for AI Software Developers as Generalist Agents [2025-ICLR] [[repo](https://github.com/All-Hands-AI/OpenHands)]
17. Alibaba LingmaAgent: Improving Automated Issue Resolution via Comprehensive Repository Exploration [2025-FSE-Companion] [[repo](https://github.com/RepoUnderstander/RepoUnderstander)]

### Smart Contract

1. ACFIX: Guiding LLMs with Mined Common RBAC Practices for Context-Aware Repair of Access Control Vulnerabilities in Smart Contracts [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.06838)]
2. Evaluating ChatGPT for Smart Contracts Vulnerability Correction [2023-COMPSAC] [[paper](https://ieeexplore.ieee.org/abstract/document/10197134)] [[repo](https://github.com/enaples/solgpt)]

### Hardware Bug

1. CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair [2024-ICLR]
2. On Hardware Security Bug Code Fixes By Prompting Large Language Models [2024-TIFS] [[paper](https://ieeexplore.ieee.org/abstract/document/10462177)] [[repo](https://zenodo.org/records/10416865)]\
   Its pre-print: Fixing Hardware Security Bugs with Large Language Models [2022-arXiv] [[paper](https://arxiv.org/abs/2302.01215)]
3. HDLdebugger: Streamlining HDL debugging with Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.11671)]
4. RTLFixer: Automatically Fixing RTL Syntax Errors with Large Language Models [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.16543)]
5. LLM4SecHW: Leveraging domain-specific large language model for hardware debugging [2023-AsianHOST] [[paper](https://arxiv.org/pdf/2401.16448)]

### Performance Bug

1. RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot [2023-arxiv] [[paper](https://arxiv.org/pdf/2306.17077)]
2. DeepDev-PERF: A Deep Learning-Based Approach for Improving Software Performance [2022-FSE/ESEC] [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549096)] [[repo](https://github.com/glGarg/DeepDev-PERF)]

### API Misuse

1. Evaluating Pre-trained Language Models for Repairing API Misuses [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.16390)] [[repo](https://anonymous.4open.science/r/TOSEM-API-Misuse)]

### Crash Bug

1. Resolving Crash Bugs via Large Language Models: An Empirical Study [2023-arxiv] [[paper](https://arxiv.org/pdf/2312.10448)] [[repo](https://chatgpt4cradiag.github.io/)]

### Test Case

1. FlakyFix: Using Large Language Models for Predicting Flaky Test Fix Categories and Test Code Repair [2024-TSE] [[repo](https://github.com/TestingResearchIllinois/idoft)]
2. NIODebugger: A Novel Approach to Repair Non-Idempotent-Outcome Tests with LLM-Based Agent [2025-ICSE] [[repo](https://github.com/zhenlongDai/)]
3. Automated Test Case Repair Using Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.06765)]
4. Identify and Update Test Cases when Production Code Changes: A Transformer-based Approach [2023-ASE]

### Error-handling Bug

1. Error Delayed Is Not Error Handled: Understanding and Fixing Propagated Error-Handling Bugs [2025-FSE/ESEC] [[repo](https://github.com/EH-Fixer/EH-Fixer)]


### Formal Proof

1. Baldur: Whole-Proof Generation and Repair with Large Language Models [2023-FSE/ESEC] [[paper](https://arxiv.org/pdf/2303.04910)]

### Translation Bug

1. Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code [2024-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3639226)] [[repo](https://github.com/Intelligent-CAT-Lab/PLTranslationEmpirical)]

### GitHub Issue

1. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? [2024-ICLR] [[paper](https://arxiv.org/pdf/2310.06770)] [[repo](https://github.com/princeton-nlp/SWE-bench)]

### Code Review

1. Divide-and-Conquer: Automating Code Revisions via Localization-and-Revision [2024-TOSEM] [[repo](https://zenodo.org/records/8373320)]
2. Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study [2024-ICSE] [[paper](https://arxiv.org/pdf/2309.08221)] [[repo](https://sites.google.com/view/chatgptcodereview)]

### Motion Planner

1. DrPlanner: Diagnosis and Repair of Motion Planners Using Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2403.07470)] [[repo](https://github.com/CommonRoad/drplanner)]


## 🙆 Human Study

1. Exploring Experiences with Automated Program Repair in Practice [2024-ICSE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3639182)]
2. Revisiting Unnaturalness for Automated Program Repair in the Era of Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.15236)] [[repo](https://zenodo.org/records/10851256)]
3. An Empirical Study of Adoption of ChatGPT for Bug Fixing among Professional Developers [2023-ITA] [[paper](https://bergersci.com/index.php/jta/article/download/19/20)]

## 🙅 Patch Correctness Assessment

1. 🔥Leveraging Large Language Model for Automatic Patch Correctness Assessment[2024-TSE] [[paper](https://ieeexplore.ieee.org/document/10659742)]
2. APPT Boosting Automated Patch Correctness Prediction via Pre-trained Language Model [2024-TSE] [[paper](https://arxiv.org/pdf/2301.12453)] [[repo](https://github.com/iSEngLab/APPT)]
3. The Best of Both Worlds: Combining Learned Embeddings with Engineered Features for Accurate Prediction of Correct Patches [2023-TOSME] [[paper](https://dl.acm.org/doi/pdf/10.1145/3576039)] [[repo](https://github.com/HaoyeTianCoder/Panther)]
4. Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning [2023-TSE] [[paper](https://arxiv.org/pdf/2301.01113)] [[repo](https://github.com/thanhlecongg/Invalidator)]
5. PatchZero: Zero-Shot Automatic Patch Correctness Assessment [2023-arxiv] [[paper](https://arxiv.org/pdf/2303.00202)]
6. Is this Change the Answer to that Problem? Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness [2021-ASE] [[paper](https://dl.acm.org/doi/pdf/10.1145/3551349.3556914)] [[repo](https://github.com/Trustworthy-Software/Quatrain)]
7. Evaluating representation learning of code changes for predicting patch correctness in program repair [2020-ASE] [[paper](https://arxiv.org/pdf/2008.02944)] [[repo](https://github.com/TruX-DTF/DL4PatchCorrectness)]

## 📊 Benchmark

1. 🔥Exploring Parameter-Efficient Fine-Tuning of Large Language Model on Automated Program Repair[2024-ASE] [[paper](https://dl.acm.org/doi/abs/10.1145/3691620.3695066)]
2. 🔥MuBench: Benchmarking Automated Program Repair: An Extensive Study on Both Real-World and Artificial Bugs [2024-ISSTA]  [[paper](https://dl.acm.org/doi/10.1145/3650212.3652140)]
3. CodeEditorBench: Evaluating Code Editing Capability of Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2404.03543)] [[repo](https://github.com/CodeEditorBench/CodeEditorBench)]
4. GitBug-Java: A Reproducible Benchmark of Recent Java Bugs [2024-arxiv] [[paper](https://arxiv.org/pdf/2402.02961)] [[repo](https://github.com/gitbugactions/gitbug-java)]
5. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? [2024-ICLR] [[paper](https://arxiv.org/pdf/2310.06770)] [[repo](https://github.com/princeton-nlp/SWE-bench)]
6. DebugBench: Evaluating Debugging Capability of Large Language Models [2024-arxiv] [[paper](https://arxiv.org/pdf/2401.04621)] [[repo](https://github.com/thunlp/DebugBench)]
7. ConDefects: A New Dataset to Address the Data Leakage Concern for LLM-based Fault Localization and Program Repair [2024-arxiv] [[paper](https://arxiv.org/pdf/2310.16253)] [[repo](https://github.com/appmlk/ConDefects)]
8. A critical review of large language model on software engineering: An example from chatgpt and automated program repair [2023-arxiv] [[paper](https://arxiv.org/pdf/2310.08879)] [[repo](https://github.com/iSEngLab/EvalGPTFix)]
9. CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation [2023-arxiv] [[paper](https://arxiv.org/pdf/2311.08588)] [[repo](https://github.com/WeixiangYAN/CodeScope)]
10. FixEval: Execution-based Evaluation of Program Fixes for Programming Problems [2023-APR] [[paper](https://arxiv.org/pdf/2206.07796)] [[repo](https://github.com/mahimanzum/FixEval)]

## 🤔 Related APR Surveys

1. A Survey of Learning-based Automated Program Repair [2023-TOSEM] [[paper](https://arxiv.org/abs/2301.03270)] [[repo](https://github.com/iSEngLab/AwesomeLearningAPR)]
2. Automatic Software Repair: A Bibliography [2018-CSUR] [paper](https://dl.acm.org/doi/10.1145/3105906)]
3. Automatic Software Repair: A Survey [2017-TSE] [paper](https://dl.acm.org/doi/10.1109/TSE.2017.2755013)]

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=iSEngLab/AwesomeLLM4APR&type=Date)](https://star-history.com/#iSEngLab/AwesomeLLM4APR&Date)




