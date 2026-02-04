# SingULAR: Single-Prompt Utility-Driven Autonomous Runtime

**SingULAR** (**SING**le-prompt **U**tility-driven **L**LM-based **A**utonomous **R**untime) is an audacious autonomous code generator designed primarily for building POCs, research prototypes, experimental workflows, scientific simulations and other excuses for using Google Colab notebooks :). It is built on top of Cursor and uses real Colab runtime execution feedback in a LangGraph agent loop to iteratively refine generated code until satisfactory results (or a hard stop) are reached - Combining the power of Cursor and Google Colab together. The main challenge this library aims to address is optimizing the iterative execution–feedback loop, with a primary focus on defining what constitutes satisfactory results and dynamically modeling a clear, testable Definition of Done (DoD) for each task.


---

## TL;DR — READ THIS BEFORE USING THE CODE

**You must read and understand this entire section before using SingULAR.**

SingULAR can operate in **two execution modes**:

#### 1. `safe-interactive`
Every code execution requires explicit user approval.  
This is the recommended mode.

#### 2. `blind-autonomous`
Execution–feedback iterations run fully unattended until convergence or a predefined stop condition. This mode is **disabled by default** and requires explicitly modifying the code to enable it, as it carries inherent security risks. 
If your project requires a large number of iterations and/or you intend to scale execution, you may consider using this mode - but read [BLIND_AUTONOMOUS_SAFETY_AND_LIABILITY.md](BLIND_AUTONOMOUS_SAFETY_AND_LIABILITY.md) first



### ⚠️ Security Model & Usage Requirements

Do not use SingULAR in any mode for malicious or adverserial purposes, cyber-attack–related prompts, or jailbreaking attempts. If you do so, you assume full responsibility for your actions; the SingULAR authors and contributors bear no responsibility or liability. If your project touches on or approaches such areas, it is your sole responsibility to carefully review all generated code before each execution. If you are unsure whether your prompt might unintentionally produce such code, err on the side of caution: review every iteration and do not use **blind-autonomous** mode.

In any mode, if you take the final output, or any intermediate artifact, produced by the code generation process outside the sterile environment for real-world use, ensure that the code is carefully reviewed before execution.

Make sure you understand SingULAR uses the Cursor API via the command-line interface and that you will be charged accordingly. Every iteration sends two calls to the cli with a prompt that is larger than the one you supplied. You can follow your usage at: https://cursor.com/dashboard?tab=usage

If you plan to use **blind-autonomous** mode, you must read [BLIND_AUTONOMOUS_SAFETY_AND_LIABILITY.md](BLIND_AUTONOMOUS_SAFETY_AND_LIABILITY.md) first. It covers the security model, sterile environment guidance, and the risks you accept by enabling unattended execution.

Also make sure you understand the **Project Status** and **Privacy & Data Handling** sections before using the code.


## How to Use:
Use this notebook https://colab.research.google.com/drive/1b0-VVeX25O4AZxZsssvEMNuFcuAfEefx?usp=sharing  it is shipped with an example prompt and solution. On the left bar go to secrets and fill your CURSOR_API_KEY, you can generate one using these instructions:

1. Log in to your **Cursor Dashboard**.
2. Navigate to the **Settings** tab (or **Integrations**, depending on your plan).
3. Locate the **User API Keys** or **Admin API Keys** section under **Advanced**.
4. Click **Create New API Key**, give it a name, and copy it immediately— it will not be shown again.


---

## Project Status

As to the current commit, SingULAR is still a WORK IN PROGRESS and *not a stable release.

- APIs may change without notice
- Behavior may be incomplete or incorrect
- Generated code should be reviewed before real-world use

**You are using this project entirely at your own risk.**

---


## Privacy & Data Handling

SingULAR is an open-source library, but it operates on top of Cursor and therefore inherits Cursor’s data-handling and privacy model.

Please be aware of the following:

1. **Prompt transmission**  
   All prompts, feedback, and generated context processed by SingULAR are ultimately **sent to Cursor** via the Cursor CLI. SingULAR itself does not intercept, store, or independently transmit this data beyond what is required for execution and logging.

2. **Cursor account settings apply**  
   How your data is stored, retained, logged, or used for model improvement is determined entirely by your Cursor account configuration and preferences, not by SingULAR. This includes any opt-in or opt-out settings related to data usage.

SingULAR does **not** provide additional privacy guarantees beyond those offered by Cursor.  
Users are responsible for reviewing Cursor’s privacy policy and configuring their Cursor settings appropriately before using SingULAR.

**Do not include secrets, private data, or sensitive information in prompts or notebook code unless you fully understand and accept the underlying data-handling policies of Cursor.**

---


## Citation

If you use SingULAR code or generated code in an academic or research publication, please cite:

```
@misc{kavvenaki2026singularity,
  title={SingULAR - Single Prompt Utility-Driven LLM-based Autonomous Runtime},
  author={Kav-Venaki, Eitam},
  year={2026},
  url={https://github.com/eitamatgithub/singularity-coder}
}
```

If relevant to your paper, please also consider citing prior work and services on execution-based coding agents:

```
@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={John Yang and Carlos E Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik R Narasimhan and Ofir Press},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```

```
@misc{cognition2024swebench,
  title={SWE-bench: Technical Report},
  author={{Cognition Labs}},
  year={2024},
  url={https://cognition.ai/blog/swe-bench-technical-report}
}
```

```
@article{wang2024openhands,
  title={OpenHands: An Open Platform for AI Software Developers as Generalist Agents},
  author={Wang, Xingyao and Li, Boxuan and Song, Yufan and Xu, Frank F. and Tang, Xiangru and Zhuge, Mingchen and Pan, Jiayi and Song, Yueqi and Li, Bowen and Singh, Jaskirat and others},
  journal={arXiv preprint arXiv:2407.16741},
  year={2024}
}
```


## Contact Details
https://www.linkedin.com/in/eitamk/


