# gbm-method

This repository includes the code used in the article *All You Need is a Paraboloid: Quadratic Cuts for Non-Convex MINLP* by A. Göß, R. Burlacu, and A. Martin (2024). 
If you use use this code in your work, please cite the article according to the corresponding section below.

A preprint is available at [https://arxiv.org/abs/2407.06143][0].


## Set-up

We encourage a set-up via an [Anaconda][1] environment.
Python version we used: Python 3.11.7

For code usage install the following packages:
- numpy (tested v1.26.4) 
- pyomo (tested v6.7.3)
- pandas (tested v2.2.2)

Numpy and pyomo are relevant for the paraboloid computation, see directory para_computation. 
For the paraboloid relaxation step for the lnts instances, pandas is additionally required in the instance extractor, see directory para_relaxation.

For the solution of the created problems, we leverage [GAMS][2]. For this, you need a valid license. As we used [Gurobi][3], you may also require a license for this. There are academic licenses for both.

As a test set we considered the MINLPLib instances that are available in OSIL format, see [instances][4]. If the you want to run the code properly, download the instances and store them in directory minlplib/osil/. 
The provided CSV-file with information about the MINLPLib instances may need to be updated.

## Usage

Note: You may need to export your PYTHONPATH variable to the repository in order to ensure proper functionality of the relative imports, e.g., `export PYTHONPATH=/path/to/repository/gbm-method`. 

In order to run the approximations by paraboloids activate the environment if one is created (`conda activate myenv`) and run `python practical_para_computation.py` from within the para_computation directory. 

For the computation regarding the lnts instances with parabolic approximations, also actiavte the environment as above and run `python write_para_relaxation.py` from within the para_relaxation directory. This writes the relaxed problem in GMS format which can be solved by gams with any available solver.
We put the resulting files for completeness.


## How to cite

As mentioned in the introducing comment, please consider the following article if you leverage the present code:
> A. Göß, R. Burlacu, and A. Martin (2024).
> All You Need is a Paraboloid: Quadratic Cuts for Non-Convex MINLP
> [https://arxiv.org/abs/2407.06143][0]

You may also use the BibTeX entry: 
```bibtex
@misc{Goess_GBM_2024,
    author={G{\"o}{\ss}, Adrian and Burlacu, Robert and Martin, Alexander},
    title={All You Need is a Paraboloid: Quadratic Cuts for Non-Convex MINLP},
    year={2024},
    eprint={2407.06143},
    archivePrefix={arXiv},
    primaryClass={math.OC},
    url={https://arxiv.org/abs/2407.06143}
}
```

[0]: https://arxiv.org/abs/2407.06143

[1]: https://docs.anaconda.com/anaconda/install/

[2]: https://www.gams.com

[3]: https://www.gurobi.com

[4]: https://www.minlplib.org/index.html
