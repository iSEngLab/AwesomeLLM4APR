# AwesomeLLM4APR: A Survey of Large Language Models for Automated Program Repair

This repository contains the materials for the paper "A Survey on Large Language Models for Automated Program Repair" by Quanjun Zhang, Chunrong Fang, and Zhenyu Chen.

## Abstract

Automated program repair (APR) aims to fix software bugs automatically. With recent advances in large language models (LLMs), there has been a surge of research leveraging LLMs for APR, demonstrating remarkable performance. This paper provides a comprehensive survey summarizing and analyzing 226 papers from 2021 to 2024 on applying LLMs to APR. It categorizes popular LLMs used for APR based on model architectures (encoder-only, decoder-only, encoder-decoder), analyzes strategies for utilizing LLMs (fine-tuning, few-shot, zero-shot), and discusses specific repair scenarios benefiting from LLMs like semantic bugs and security vulnerabilities. Critical aspects including input prompting, open science issues, industrial adoption, and available tools are covered. Finally, open challenges and potential future research directions are highlighted.

## Contents

- [Survey Methodology](#survey-methodology)
- [Publication Trends](#publication-trends)
- [Popular LLMs for APR](#popular-llms-for-apr)
- [LLM Utilization Strategies](#llm-utilization-strategies)
- [Repair Scenarios](#repair-scenarios)
- [Other Critical Aspects](#other-critical-aspects)  
- [Challenges and Future Directions](#challenges-and-future-directions)
- [Citation](#citation)

## Survey Methodology

The authors followed a systematic literature review methodology to collect and analyze relevant papers, detailing the research questions, search strategy, study selection criteria and process.

## Publication Trends 

Analysis of the distribution of publication years, venues, contribution types (new techniques, empirical studies, benchmarks) for LLM-based APR research.

## Popular LLMs for APR

Popular LLMs applied to APR are categorized into encoder-only (e.g. CodeBERT), decoder-only (e.g. CodeGen, GPT models), and encoder-decoder (e.g. CodeT5) based on their architectures.  

## LLM Utilization Strategies  

The three main strategies for utilizing LLMs are discussed:
1. **Fine-tuning**: Further training LLMs on bug fix data
2. **Few-shot learning**: Using task examples to guide LLMs  
3. **Zero-shot learning**: Direct prompting of LLMs without examples

## Repair Scenarios

Application of LLMs to different repair scenarios is analyzed, with semantic bugs receiving most attention, followed by security vulnerabilities, syntax errors, static warnings and others.

## Other Critical Aspects

Other important aspects covered include:
- Input prompt engineering techniques
- Open science and publicly available artifacts 
- Industrial adoption of LLM-based APR
- Available tools and prototypes 

## Challenges and Future Directions  

Current open challenges in applying LLMs to APR are discussed, along with potential future research directions.

## Citation

```bibtex
@article{zhang2024survey,
  title={A Survey on Large Language Models for Automated Program Repair},
  author={Zhang, Quanjun and Fang, Chunrong and Chen, Zhenyu},
  journal={ACM Computing Surveys},
  year={2024}
}
```

## Support

If you have any questions, please contact Chunrong Fang (fangchunrong@nju.edu.cn).

Let me know if you need any clarification or would like me to modify anything in this draft.

## LLM4SE
| Task | Paper Title | Year | Publisher |
| --- | --- | --- | --- |
| Program Repair | [TFix-Learning to Fix Coding Errors with a Text-to-Text Transformer](https://github.com/eth-sri/TFix) | 2021 | PMLR |
| Program Repair | Generating Bug-Fixes Using Pretrained Transformers | 2021 | PLDI |
| Program Repair | [Applying CodeBERT for Automated Program Repair of Java Simple Bugs](https://github.com/EhsanMashhadi/MSR2021-ProgramRepair) | 2021 | MSR |
| Vulnerability Repair | [VulRepair: A T5-Based Automated Software Vulnerability Repair](https://github.com/awsm-research/VulRepair) | 2022 | FSE/ESEC |
| Program Repair | [CIRCLE: Continual Repair across Programming Languages](https://github.com/2022CIRCLE/CIRCLE) | 2022 | ISSTA |
| Program Repair | An empirical study of deep transfer learning-based program repair for Kotlin projects | 2022 | FSE/ESEC |
| Code Edits Prediction | [DeepDev-PERF: A Deep Learning-Based Approach for Improving Software Performance](https://github.com/glGarg/DeepDev-PERF) | 2022 | FSE/ESEC |
| Code Completion | [SYNSHINE: improved fixing of Syntax Errors](https://zenodo.org/record/4572390#.Y4CY8xRByUk) | 2022 | TSE |
| Program Repair | [Towards JavaScript program repair with Generative Pre-trained Transformer (GPT-2)](https://github.com/AAI-USZ/APR22-JS-GPT) | 2022 | APR |
| Program Repair | Repairing bugs in python assignments using large language models | 2022 | arxiv |
| Program Repair | Fix Bugs with Transformer through a Neural-Symbolic Edit Grammar | 2022 | ICLR |
| Program Repair | Patch Generation with Language Models: Feasibility and Scaling Behavior | 2022 | ICLR |
| Program Repair | Can OpenAI's codex fix bugs?: an evaluation on QuixBugs | 2022 | APR |
| Program Repair | [An Analysis of the Automatic Bug Fixing Performance of ChatGPT](https://gitlab.rlp.net/dsobania/chatgpt-apr) | 2022 | APR |
| Program Repair | [Less training, more repairing please: revisiting automated program repair via zero-shot learning](https://zenodo.org/records/6819444) | 2022 | FSE/ESEC |
| Program Repair | [Framing Program Repair as Code Completion](https://github.com/FranciscoRibeiro/code-truncater) | 2022 | APR |
| Program Repair | [Automated Program Repair in the Era of Large Pre-trained Language Models](https://zenodo.org/records/7592886) | 2023 | ICSE |
| Program Repair | Repair Is Nearly Generation: Multilingual Program Repair with LLMs | 2023 | AAAI |
| Code Completion | [Retrieval-based prompt selection for code-related few-shot learning](https://github.com/prompt-learning/cedar) | 2023 | ICSE |
| Code Completion | [What makes good in-context demonstrations for code intelligence tasks with llms?](https://github.com/shuzhenggao/ICL4code) | 2023 | ASE |
| Vulnerability Repair | [Examining zero-shot vulnerability repair with large language models](https://drive.google.com/drive/folders/1xJ-z2Wvvg7JSaxfTQdxayXFEmoF3y0ET?usp=sharing) | 2023 | S&P |
| Vulnerability Repair | [Evaluating ChatGPT for Smart Contracts Vulnerability Correction](https://github.com/enaples/solgpt) | 2023 | COMPSAC |
| Program Synthesis | [Fully Autonomous Programming with Large Language Models](https://zenodo.org/records/7837282) | 2023 | GECCO |
| Program Repair | [Automated Program Repair Using Generative Models for Code Infilling](https://github.com/KoutchemeCharles/aied2023) | 2023 | AIED |
| Code Completion | Fixing Rust Compilation Errors using LLMs | 2023 | arxiv |
| Program Repair | STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing | 2023 | arxiv |
| Program Repair | Conversational automated program repair | 2023 | arxiv |
| Program Repair | [Is ChatGPT the Ultimate Programming Assistant--How far is it?](https://github.com/HaoyeTianCoder/ChatGPT-Study) | 2023 | arxiv |
| Bug Localization | Using Large Language Models for Bug Localization and Fixing | 2023 | iCAST |
| Code Summarization | [SkipAnalyzer: An Embodied Agent for Code Analysis with Large Language Models](https://zenodo.org/records/10043170) | 2023 | arxiv |
| Program Repair | [An Empirical Study on Fine-Tuning Large Language Models of Code for Automated Program Repair](https://github.com/LLMC-APR/STUDY) | 2023 | ASE |
| Program Repair | An Evaluation of the Effectiveness of OpenAI's ChatGPT for Automated Python Program Bug Fixing using QuixBugs | 2023 | iSEMANTIC |
| Program Repair | [Keep the Conversation Going: Fixing 162 out of 337 bugs for $0.42 each using ChatGPT](https://arxiv.org/abs/2303.17193) | 2023 | arxiv |
| Debugging | Explainable Automated Debugging via Large Language Model-driven Scientific Debugging | 2023 | arxiv |
| Vulnerability Repair | A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification | 2023 | arxiv |
| Code Completion | [A Chain of AI-based Solutions for Resolving FQNs and Fixing Syntax Errors in Partial Code](https://github.com/SE-qinghuang/A-Chain-of-AI-based-Solutions-for-Resolving-FQNs-and-Fixing-Syntax-Errors-in-Partial-Code) | 2023 | arxiv |
| Debugging | [GPT-3-Powered Type Error Debugging: Investigating the Use of Large Language Models for Code Repair](https://gitlab.com/FranciscoRibeiro/mentat) | 2023 | SLE |
| Bug Localization | [Resolving Crash Bugs via Large Language Models: An Empirical Study](https://chatgpt4cradiag.github.io/) | 2023 | arxiv |
| Code Review | The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model | 2023 | arxiv |
| Vulnerability Repair | [Fixing Hardware Security Bugs with Large Language Models](https://zenodo.org/records/7540216) | 2023 | arxiv |
| Program Repair | [Impact of Code Language Models on Automated Program Repair](https://github.com/lin-tan/clm) | 2023 | ICSE |
| Code Generation | Towards Generating Functionally Correct Code Edits from Natural Language Issue Descriptions | 2023 | arxiv |
| Code Edits Prediction | [The Plastic Surgery Hypothesis in the Era of Large Language Models](https://arxiv.org/abs/2303.10494) | 2023 | ASE |
| Vulnerability Detection | Exploring the Limits of ChatGPT in Software Security Applications | 2023 | arxiv |
| Vulnerability Repair | ZeroLeak: Using LLMs for Scalable and Cost Effective Side-Channel Patching | 2023 | arxiv |
| Code Understanding | [CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation](https://github.com/WeixiangYAN/CodeScope) | 2023 | arxiv |
| Code Review | [How ChatGPT is Solving Vulnerability Management Problem](https://anonymous.4open.science/r/DefectManagementEvaluation-0411) | 2023 | arxiv |
| GUI Testing | Guiding ChatGPT to Fix Web UI Tests via Explanation-Consistency Checking | 2023 | arxiv |
| Vulnerability Repair | [How Effective Are Neural Networks for Fixing Security Vulnerabilities](https://github.com/lin-tan/llm-vul) | 2023 | ISSTA |
| Program Repair | [Enhancing Automated Program Repair through Fine-tuning and Prompt Engineering](https://zenodo.org/records/8122636) | 2023 | arxiv |
| Program Repair | [Training Language Models for Programming Feedback Using Automated Repair Tools](https://github.com/KoutchemeCharles/aied2023) | 2023 | AIED |
| Program Repair | [RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair](https://anonymous.4open.science/r/repairllama-BC13) | 2023 | arxiv |
| Code Generation | [Automated Code Editing with Search-Generate-Modify](https://github.com/SarGAMTEAM/SarGAM.git) | 2023 | arxiv |
| Code Generation | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://www.swebench.com/) | 2023 | arxiv |
| Vulnerability Repair | [Pre-trained Model-based Automated Software Vulnerability Repair: How Far are We?](https://github.com/iSEngLab/LLM4VulFix) | 2023 | TDSC |
| Program Repair | [RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair](https://figshare.com/s/a4e95baee01bba14bf4b) | 2023 | FSE/ESEC |
| Program Repair | [InferFix: End-to-End Program Repair with LLMs over Retrieval-Augmented Prompts](https://github.com/microsoft/InferredBugs) | 2023 | FSE/ESEC |
| Program Repair | [A critical review of large language model on software engineering: An example from chatgpt and automated program repair](https://github.com/iSEngLab/EvalGPTFix) | 2023 | arxiv |
| Program Repair | Neural Program Repair with Program Dependence Analysis and Effective Filter Mechanism | 2023 | arxiv |
| Program Repair | [Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback](https://github.com/Lune-Blue/COFFEE) | 2023 | arxiv |
| Program Repair | A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair | 2023 | arxiv |
| Program Repair | [Automated Repair of Programs from Large Language Models](https://github.com/zhiyufan/apr4codex) | 2023 | ICSE |
| Program Repair | [Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair](https://github.com/ise-uiuc/Repilot) | 2023 | FSE/ESEC |
| Program Repair | [Gamma: Revisiting Template-Based Automated Program Repair Via Mask Prediction](https://github.com/iSEngLab/GAMMA) | 2023 | ASE |
| Program Repair | STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing | 2023 | arxiv | 
| Code Edits Prediction | RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot | 2023 | arxiv |
| Program Repair | [An Extensive Study on Model Architecture and Program Representation in the Domain of Learning-based Automated Program Repair](https://github.com/AAI-USZ/APR23-representations) | 2023 | APR |
| Vulnerability Repair | [Vision Transformer-Inspired Automated Vulnerability Repair](https://github.com/awsm-research/VQM) | 2023 | TOSEM |
| Program Repair | [Improving Automated Program Repair with Domain Adaptation](https://github.com/arminzirak/TFix) | 2023 | TOSEM |
| Test Generation | [FixEval: Execution-based Evaluation of Program Fixes for Programming Problems](https://github.com/mahimanzum/FixEval) | 2023 | APR |
| Program Repair | [A Novel Approach for Automatic Program Repair using Round-Trip Translation with Large Language Models](https://zenodo.org/records/10500594) | 2024 | arxiv |
| Code Edits Prediction | [Frustrated with Code Quality Issues? LLMs can Help!](https://aka.ms/CORE_MSRI) | 2024 | FSE/ESEC |
| Code Completion | [Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors](https://github.com/JohnnyPeng18/TypeFix) | 2024 | ICSE |
| Code Completion | [PyTy: Repairing Static Type Errors in Python](https://github.com/sola-st/PyTy) | 2024 | ICSE |
| Vulnerability Repair | Enhanced Automated Code Vulnerability Repair using Large Language Models | 2024 | arxiv |
| Program Repair | [Out of Context: How important is Local Context in Neural Program Repair?](https://github.com/giganticode/out_of_context_paper_data) | 2024 | ICSE |
| Debugging | [DebugBench: Evaluating Debugging Capability of Large Language Models](https://github.com/thunlp/DebugBench) | 2024 | arxiv |
| Vulnerability Repair | [Large Language Model as Synthesizer: Fusing Diverse Inputs for Better Automatic Vulnerability Repair](https://github.com/soarsmu/VulMaster_) | 2024 | ICSE |
| Requirements Classification | [Evaluating Pre-trained Language Models for Repairing API Misuses](https://anonymous.4open.science/r/TOSEM-API-Misuse) | 2023 | arxiv |
| Program Repair | [CURE Code-Aware Neural Machine Translation for Automatic Program Repair](https://github.com/lin-tan/CURE) | 2021 | ICSE |
| Program Repair | [DEAR A Novel Deep Learning-based Approach for Automated Program Repair](https://github.com/AutomatedProgramRepair-2021/dear-auto-fix) | 2022 | ICSE |