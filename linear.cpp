#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include <set>
#include <map>
#include <vector>
#include <algorithm>

#ifndef TIMER
#define TIMER
#define NS_PER_SEC 1000000000
double wall_time_diff(int64_t ed, int64_t st)
{
	return (double)(ed-st)/(double)NS_PER_SEC;
}

int64_t wall_clock_ns()
{
#if __unix__
	struct timespec tspec;
	clock_gettime(CLOCK_MONOTONIC, &tspec);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
#if __MACH__
	return 0;
#else
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
#endif
}
#endif

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
class sparse_operator
{
public:
	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);
extern int dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern int dgemm_(char *, char*, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void dgetrf_(int *, int *, double *, int *, int *, int *);
extern void dgetri_(int *, double *, int *, int *, double *, int *, int *);

#ifdef __cplusplus
}
#endif
void inverse(double* A, int N) //compute the inverse of a symmetric PD matrix
{
	int *IPIV = new int[N+1];
	int LWORK = N*N;
	double *WORK = new double[LWORK];
	int INFO;

	dgetrf_(&N,&N,A,&N,IPIV,&INFO);
	dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

	delete[] IPIV;
	delete[] WORK;
}

class regularized_fun
{
public:
	/* functions for general proximal variable metric method */
	virtual double fun(double *w) = 0; //Evaluate the objective function
	virtual void loss_grad(double *w, double *g) = 0; //Evaluate the gradient of the smooth part. w is the input and g is the output
	virtual int setselection(double *w, double *loss_g, int *index) = 0; //Set selection to accelerate the subproblem solve. index[0] to index[return_value-1] are the coordinates selected
	virtual int get_nr_variable(void) = 0; //Return problem dimension
	virtual void prox_grad(double *w, double *g, double *step, double *oldd, double alpha = 0, int length = -1) = 0; //Implements step = Prox_{regularizer}(w + oldd - g / alpha) and all are of dimension length
	virtual double regularizer(double *w, int length) = 0; //Evaluate regularizer(w) of dimension length

	/* functions for IPVM+: smooth Newton step related functions */
	virtual void full_grad(double *w, double *g, int *index, int index_length, double *full_g, std::vector<int> &nonzero_set) = 0; //gradient of both parts when restricted to the current active smooth manifold
	virtual void full_Hv(double *s, double *Hs, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL) = 0; //Hessian-vector of both terms
	virtual void get_diag_preconditioner(double *M, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL) = 0; //Get the diagonal entries of the Hessian of the regularized objective restricted to the current active manifold
	virtual double vHv(double *v, int *index = NULL, int index_length = -1) = 0; //For estimating the starting step size for LBFGS when m = 0
	virtual double smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew) = 0; //Line search within the smooth manifold


	/* functions for proximal Newton: related to nabla^2 f */
	virtual void diagH(double *M, const int *index = NULL, int index_length = -1) = 0; //Comute the diagonal entries of the Hessian of the smooth part
	virtual int get_intermediate_size(void) = 0; //Get the length of the intermediate vector for easier computation of the gradient and Hessian-vector products in the smooth part
	virtual double subHv(double *intermediate, int idx) = 0; //Compute Hessian-vector product of a single coordinate given the intermediate vector
	virtual void update_intermediate(double *intermediate, double ddiff, int idx) = 0; //Update the intermediate vector when one entry is updated
	virtual double vHv_from_intermediate(double *intermediate) = 0; //Compute vHv from the intermediate vector

	/* For the trust-region-like approach */
	virtual double f_update(double *w, double *step, double Q, double eta, int *index = NULL, int index_length = -1) = 0; //Update w to w + step and return the new objective. It is updated if F(w + step) <= F(w) + eta * Q and remains unchanged otherwise. index specifies the coordinates of the dense array step

protected:
	const problem *prob;
	double reg;
	double current_f;
};

class l1r_fun: public regularized_fun
{
public:
	l1r_fun(const problem *prob, double C);
	virtual ~l1r_fun(){}

	/* functions for general proximal variable metric method */
	double fun(double *w);
	void loss_grad(double *w, double *g);
	int setselection(double *w, double *loss_g, int *index);
	int get_nr_variable(void);
	void prox_grad(double *w, double *g, double *step, double *oldd, double alpha = 0, int length = -1);
	double regularizer(double *w, int length);
	
	/* functions for IPVM+: smooth Newton step related functions */
	void full_grad(double *w, double *g, int *index, int index_length, double *full_g, std::vector<int> &nonzero_set);
	void full_Hv(double *s, double *Hs, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL);
	void get_diag_preconditioner(double *M, const std::vector<int> &fullindex = std::vector<int>{-1}, double *w = NULL);
	double vHv(double *v, int *index = NULL, int index_length = -1); //For estimating the starting step size in sparsa
	double smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew);

	/* functions for proximal Newton: related to nabla^2 f */
	void diagH(double *M, const int *index = NULL, int index_length = -1);
	int get_intermediate_size(void);
	double subHv(double *intermediate, int idx);
	void update_intermediate(double *intermediate, double ddiff, int idx);
	double vHv_from_intermediate(double *intermediate);

	/* For the trust-region-like approach */
	double f_update(double *w, double *step, double Q, double eta, int *index = NULL, int index_length = 0);

protected:
	virtual double loss(int i, double wx_i) = 0;
	virtual void update_zD() = 0;
	void Xv(double *v, double *Xv, const int *index = NULL, int index_length = -1);
	void XTv(double *v, double *XTv, const int *index = NULL, int index_length = -1);

	double C;
	double *z;
	double *Xw;
	double *D;
};

class l1r_lr_fun: public l1r_fun
{
public:
	l1r_lr_fun(const problem *prob, double C);
	~l1r_lr_fun();
	
protected:
	double loss(int i, double xw_i);
	void update_zD();
};

class lasso: public l1r_lr_fun
{
public:
	lasso(const problem *prob, double C);
	~lasso();

private:
	double loss(int i, double xw_i); // Loss is (w^T x - y)^2 / 2
	void update_zD();

};

class l1r_l2_svc_fun: public l1r_lr_fun
{
public:
	l1r_l2_svc_fun(const problem *prob, double C);
	~l1r_l2_svc_fun();

private:
	double loss(int i, double xw_i);
	void update_zD();

};

l1r_fun::l1r_fun(const problem *prob, double C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	Xw = new double[l];
	D = new double[l];
	this->C = C;
}

double l1r_fun::regularizer(double *w, int length)
{
	double ret = 0;
	for (int i=0;i<length;i++)
		ret += fabs(w[i]);
	return ret;
}

double l1r_fun::fun(double *w)
{
	int i;
	double f=0;
	int l=prob->l;
	int w_size = get_nr_variable();
	int nnz = 0;
	for (i=0;i<w_size;i++)
		nnz += (w[i] != 0);
	if (nnz == 0)
		memset(Xw, 0, sizeof(double) * l);
	else if ((double)nnz / w_size < 0.5)
	{
		int *index = new int[nnz];
		double *subw = new double[nnz];
		int counter = 0;
		for (i=0;i<w_size;i++)
			if (w[i] != 0)
			{
				subw[counter] = w[i];
				index[counter++] = i;
			}

		Xv(subw, Xw, index, nnz);
		delete[] index;
		delete[] subw;
	}
	else
		Xv(w, Xw);
	reg = regularizer(w, w_size);
	for(i=0;i<l;i++)
		f += loss(i, Xw[i]);
	f = C * f + reg;

	current_f = f;
	return(f);
}

double l1r_fun::f_update(double *w, double *step, double Q, double eta, int *index, int index_length)
{
	if (Q > 0)
		return current_f;
	int i;
	int inc = 1;
	int l = prob->l;
	double one = 1.0;
	double *substep;
	int *subindex;
	int nnz = 0;
	int len = index_length;
	int w_size = get_nr_variable();
	if (len < 0)
		len = w_size;


	double reg_diff = 0;

	if (index != NULL) // If a subset is selected
		for (i=0;i<index_length;i++)
			reg_diff += fabs(w[index[i]] + step[i]) - fabs(w[index[i]]);
	else
		for (i=0;i<w_size;i++)
			reg_diff += fabs(w[i] + step[i]) - fabs(w[i]);

	if (index_length < 0)
		Xv(step, z);
	else
	{
		for (i=0;i<len;i++)
			nnz += (step[i] != 0);
		if (nnz == 0)
			memset(z, 0, sizeof(double) * l);
		else if (nnz < len * 0.5)
		{
			substep = new double[nnz];
			subindex = new int[nnz];
			int counter = 0;
			if (index == NULL)
			{
				for (i=0;i<w_size;i++)
					if (step[i] != 0)
					{
						substep[counter] = step[i];
						subindex[counter++] = i;
					}
			}
			else
			{
				for (i=0;i<len;i++)
					if (step[i] != 0)
					{
						substep[counter] = step[i];
						subindex[counter++] = index[i];
					}
			}
			Xv(substep, z, subindex, nnz);
			delete[] substep;
			delete[] subindex;
		}
		else
			Xv(step, z, index, index_length);
	}


	daxpy_(&l, &one, z, &inc, Xw, &inc);
	double f_new = 0;
	for(i=0; i<l; i++)
		f_new += loss(i, Xw[i]);
	f_new *= C;
	f_new += reg_diff + reg;
	if (f_new - current_f <= eta * Q)
	{
		current_f = f_new;
		reg += reg_diff;
	}
	else
	{
		double factor = -1;
		daxpy_(&l, &factor, z, &inc, Xw, &inc);
	}
	return current_f;
}

void l1r_fun::loss_grad(double *w, double *g)
{
	update_zD();

	XTv(z, g);
}

void l1r_fun::full_Hv(double *s, double *Hs, const std::vector<int> &fullindex, double *w)
{
	int i;
	int l=prob->l;
	int index_length = (int) fullindex.size();
	const int *index= &fullindex[0];
	Xv(s, z, index, index_length);
	for(i=0;i<l;i++)
		z[i] = C*D[i]*z[i];

	XTv(z, Hs, index, index_length);
}

double l1r_fun::vHv(double *s, int *index, int index_length)
{
	int i;
	int inc = 1;
	int l=prob->l;
	bool dense = (index_length < 0 || index_length == get_nr_variable());
	double *subs;

	if (dense)
		Xv(s, z);
	else
	{
		subs = new double[index_length];
		for (i=0;i<index_length;i++)
			subs[i] = s[index[i]];
		Xv(subs, z, index, index_length);
	}

	double alpha = 0;
	for(i=0;i<l;i++)
		alpha += z[i] * z[i] * D[i];// v^THv = C (Xv)^T D (Xv)
	double norm2;
	if (dense)
	{
		int w_size = get_nr_variable();
		norm2 = ddot_(&w_size, s, &inc, s, &inc);
	}
	else
		norm2 = ddot_(&index_length, subs, &inc, subs, &inc);
	alpha *= C / norm2;
	return alpha;
}

void l1r_fun::get_diag_preconditioner(double *M, const std::vector<int> &fullindex, double *w)
{
	diagH(M, &fullindex[0], (int)fullindex.size());
}

void l1r_fun::diagH(double *M, const int *index, int index_length)
{
	int i;
	int w_size=index_length;
	if (w_size < 0)
		w_size = get_nr_variable();
	feature_node **x = prob->x;
	memset(M, 0, sizeof(double) * w_size);

	for (i=0; i<w_size; i++)
	{
		feature_node *s;
		s=x[index[i]];
		while (s->index!=-1)
		{
			M[i] += s->value*s->value*C*D[s->index - 1];
			s++;
		}
	}
}

void l1r_fun::full_grad(double *w, double *g, int *index, int index_length, double *full_g, std::vector<int> &nonzero_set)
{
	for (int i=0, full_i=0;i<index_length;i++)
	{
		int j = index[i];
		if (w[j] != 0)
		{
			nonzero_set.push_back(j);
			full_g[full_i++] = g[j] + (2 * (w[j] > 0) - 1);
		}
	}
}

void l1r_fun::Xv(double *v, double *Xv, const int *index, int index_length)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	if (index_length >= 0)
		w_size = index_length;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=0;
	for(i=0;i<w_size;i++)
		if (v[i] != 0)
		{
			feature_node *s;
			if (index_length >= 0)
				s = x[index[i]];
			else
				s = x[i];
			sparse_operator::axpy(v[i], s, Xv);
		}
}

void l1r_fun::XTv(double *v, double *XTv, const int *index, int index_length)
{
	int i;
	int w_size=get_nr_variable();
	if (index_length >= 0)
		w_size = index_length;
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
	{
		feature_node * s;
		if (index_length < 0)
			s=x[i];
		else
			s=x[index[i]];
		XTv[i] = sparse_operator::dot(v, s);
	}
}

int l1r_fun::setselection(double *w, double *loss_g, int *index)
{
	int i;
	int index_length = 0;
	int w_size = get_nr_variable();

	for (i=0; i<w_size; i++)
		if (w[i] != 0 || loss_g[i] < -1 || loss_g[i] > 1)
		{
			index[index_length] = i;
			index_length++;
		}
	return index_length;
}

int l1r_fun::get_nr_variable(void)
{
	return prob->n;
}

int l1r_fun::get_intermediate_size(void)
{
	return prob->l;
}

double l1r_fun::smooth_line_search(double *w, double *smooth_step, double delta, double eta, std::vector<int> &index, double *fnew)
{
	double discard_threshold = 1e-4; //If the step size is too small then do not take this step
	double discard_threshold2 = 1e-6; //If the step size is too small then do not take this step
	int i;
	int l = prob->l;
	int inc = 1;
	double step_size = 1;
	int max_num_linesearch = 10;
	for (i=0;i<(int)index.size();i++)
		if (smooth_step[i] != 0)
		{
			double tmp = -w[index[i]] / smooth_step[i];
			if (tmp > 0)
				step_size = min(step_size, tmp);
		}
	if (step_size < discard_threshold)
	{
		info("INITIAL STEP SIZE TOO SMALL: %g\n",step_size);
		*fnew = current_f;
		return -1;
	}


	double reg_diff = 0;
	for (i=0;i<(int) index.size();i++)
		if (smooth_step[i] != 0)
			reg_diff += fabs(w[index[i]] + step_size * smooth_step[i]) - fabs(w[index[i]]);
	reg_diff /= step_size;

	Xv(smooth_step, z, &index[0], (int) index.size());

	int num_linesearch;
	delta *= eta;

	daxpy_(&l, &step_size, z, &inc, Xw, &inc);
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		if (step_size < discard_threshold2)
		{
			info("LINE SEARCH FAILED: Step size = %g\n",step_size);
			double factor = -step_size;
			daxpy_(&l, &factor, z, &inc, Xw, &inc);
			step_size = 0;
			*fnew = current_f;
			break;
		}
		double f_new = 0;
		for(i=0; i<l; i++)
			f_new += loss(i, Xw[i]);
		//If profiling gets sparsity, should consider tracking loss as a sum and only update individual losses
		f_new = f_new * C + reg + reg_diff * step_size;
		if (f_new - current_f <= delta * step_size)
		{
			current_f = f_new;
			*fnew = f_new;
			reg += reg_diff * step_size;
			break;
		}
		else
		{
			step_size *= 0.5;
			double factor = -step_size;
			daxpy_(&l, &factor, z, &inc, Xw, &inc);
		}
	}
	if (num_linesearch == max_num_linesearch)
	{
		info("LINE SEARCH FAILED: Step size = %g\n",step_size);
		double factor = -step_size;
		daxpy_(&l, &factor, z, &inc, Xw, &inc);
		step_size = 0;
		*fnew = current_f;
	}

	return step_size;
}

//step = prox_reg(w + oldd - g / alpha)
void l1r_fun::prox_grad(double *w, double *g, double *local_step, double *oldd, double alpha, int length)
{
	if (alpha <= 0)
	{
		int inc = 1;
		alpha = ddot_(&length, g, &inc, g, &inc);
		alpha = sqrt(alpha);
	}
	if (length < 0)
		length = get_nr_variable();
	for (int i=0;i<length;i++)
	{
		double u = w[i] + oldd[i] - g[i] / alpha;
		local_step[i] = ((u > 0) - (u<0)) * max(fabs(u) - 1.0 / alpha, 0.0);
	}
}

double l1r_fun::subHv(double *intermediate, int idx)
{
	double ret = 0;
	feature_node *x = prob->x[idx];
	while(x->index != -1)
	{
		int idx2 = x->index-1;
		ret += D[idx2]*intermediate[idx2]*x->value;
		x++;
	}
	return (C*ret);
}

double l1r_fun::vHv_from_intermediate(double *intermediate)
{
	double ret = 0;
	for (int i=0;i<prob->l;i++)
		ret += D[i] * intermediate[i] * intermediate[i];
	return (C*ret);
}

void l1r_fun::update_intermediate(double *intermediate, double ddiff, int idx)
{
	sparse_operator::axpy(ddiff, prob->x[idx], intermediate);
}

l1r_lr_fun::l1r_lr_fun(const problem *prob, double C): l1r_fun(prob, C)
{
}

l1r_lr_fun::~l1r_lr_fun()
{
	delete[] z;
	delete[] Xw;
	delete[] D;
}

double l1r_lr_fun::loss(int i, double xw_i)
{
	double yXw = prob->y[i]*xw_i;
	if (yXw >= 0)
		return log(1 + exp(-yXw));
	else
		return (-yXw+log(1 + exp(yXw)));
}

void l1r_lr_fun::update_zD()
{
	for(int i=0;i<prob->l;i++)
	{
		z[i] = 1/(1 + exp(-prob->y[i]*Xw[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C*(z[i]-1)*prob->y[i];
	}
}


lasso::lasso(const problem *prob, double C): l1r_lr_fun(prob, C)
{
	for (int i=0;i<prob->l;i++)
		D[i] = 1.0;
}

lasso::~lasso()
{
}

double lasso::loss(int i, double xw_i)
{
	double tmp = (prob->y[i] - xw_i);
	return tmp * tmp / 2;
}

void lasso::update_zD()
{
	for(int i=0;i<prob->l;i++)
		z[i] = Xw[i] - prob->y[i];
}

l1r_l2_svc_fun::l1r_l2_svc_fun(const problem *prob, double C):
	l1r_lr_fun(prob, C)
{
}

l1r_l2_svc_fun::~l1r_l2_svc_fun()
{
}

double l1r_l2_svc_fun::loss(int i, double wx_i)
{
	double d = 1 - prob->y[i] * wx_i;
	if (d > 0)
		return C * d * d;
	else
		return 0;
}

void l1r_l2_svc_fun::update_zD()
{
	for(int i=0;i<prob->l;i++)
	{
		z[i] = Xw[i] * prob->y[i];
		if (z[i] < 1)
		{
			z[i] = 2 * C*prob->y[i]*(z[i]-1);
			D[i] = 2;
		}
		else
		{
			z[i] = 0;
			D[i] = 0;
		}
	}
}

// begin of solvers
void fill_range(std::vector<int> &v, int n)
{
	v.clear();
	for (int i = 0; i < n; i++)
		v.push_back(i);
}


class SOLVER
{
public:
	SOLVER(const regularized_fun *fun_obj, double eps = 0.1, double eta = 1e-4, int max_iter = 10000);
	~SOLVER();
	void set_print_string(void (*i_print) (const char *buf));

protected:
	double eps;
	double eta;
	int max_iter;
	regularized_fun *fun_obj;
	void info(const char *fmt,...);
	void (*solver_print_string)(const char *buf);
};

class IPVM_LBFGS: public SOLVER
{
public:
	IPVM_LBFGS(const regularized_fun *fun_obj, double eps = 0.0001, int m=10, double eta = 1e-4, int max_iter = 1000000);
	~IPVM_LBFGS();

	void ipvm_lbfgs(double *w);
protected:
	int solver;
	int M;
	double inner_eps;
	int max_inner;
	int min_inner;
	double sparse_factor;
	double update_eps;
	void update_inner_products(double **inner_product_matrix, int k, int DynamicM, double *s, double *y, int *index = NULL, int index_length = -1);
	void compute_R(double *R, int DynamicM, double **inner_product_matrix, int k, double gamma);
	double rpcd(double *w, double *loss_g, double *R, double *s, double *y, double gamma, double *step, double scaling, int DynamicM, int *index, int index_length);
	double Q_prox_grad(double *step, double *w, double *g, int index_length, int *index, double alpha, double *tmps, double *f, int *counter_ret);
	double newton(double *g, double *step, const std::vector<int> &nonzero_set, int max_cg_iter, double *w = NULL);
};

class IPVM_NEWTON: public IPVM_LBFGS
{
public:
	IPVM_NEWTON(const regularized_fun *fun_obj, double eps = 0.0001, double eta = 1e-4, int max_iter = 1000000);
	~IPVM_NEWTON();

	void ipvm_newton(double *w);
protected:
	double rpcd(double *w, double *loss_g, double *step, double damping, double scaling, int *index, int index_length);
};

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void SOLVER::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*solver_print_string)(buf);
}

void SOLVER::set_print_string(void (*print_string) (const char *buf))
{
	solver_print_string = print_string;
}

SOLVER::SOLVER(const regularized_fun *fun_obj, double eps, double eta, int max_iter)
{
	this->fun_obj=const_cast<regularized_fun *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->eta = eta;
	solver_print_string = default_print;
}

SOLVER::~SOLVER()
{
}


enum Stage {initial, alternating, smooth};

IPVM_LBFGS::IPVM_LBFGS(const regularized_fun *fun_obj, double eps, int m, double eta, int max_iter):
	SOLVER(fun_obj, eps, eta, max_iter)
{
	this->solver = solver;
	this->M = m;
	this->inner_eps = inner_eps;
	this->max_inner = 100;
	this->min_inner = 5;
	update_eps = 1e-10;//Ensure PD of the LBFGS matrix
	sparse_factor = 0.2;//The sparsity threshold for deciding when we should switch to sparse operations
}

IPVM_LBFGS::~IPVM_LBFGS()
{
}

void IPVM_LBFGS::ipvm_lbfgs(double *w)
{
	const int max_modify_Q = 20;
	const int smooth_trigger = 10;
	const double ALPHA_MAX = 1e10;
	const double ALPHA_MIN = 1e-4;
	//const int switch_threshold = M;
	int inc = 1;
	double one = 1.0;
	double minus_one = -1.0;

	int n = fun_obj->get_nr_variable();

	int i, k = 0;
	int iter = 0;
	int skip = 0;
	int DynamicM = 0;
	double s0y0 = 0;
	double s0s0 = 0;
	double y0y0 = 0;
	double f;
	double Q0;
	double Q = 0;
	double gamma = ALPHA_MAX;
	int skip_flag;
	int64_t timer_st, timer_ed;
	double accumulated_time = 0;
	double alpha;

	double *s = new double[2*M*n];
	double *y = s + M*n;
	double *tmpy = new double[n];
	double *tmps = new double[n];
	double *loss_g = new double[n];
	double *step = new double[n];
	double *full_g = new double[n];
	double *R = new double[4 * M * M];
	double **inner_product_matrix = new double*[M];
	double fnew;
	int counter = 0;
	int ran_smooth_flag = 0;
	Stage stage = initial;
	int latest_set_size = 0;
	int init_max_cg_iter = 5;
	double cg_increment = 2;
	int current_max_cg_iter = init_max_cg_iter;
	int search = 1;

	for (i=0; i < M; i++)
		inner_product_matrix[i] = new double[2*M];

	// for alternating between previous and current indices
	int *index = new int[n];
	int index_length = n;
	int unchanged_counter = 0;

	// calculate the Q quantity used in update acceptance with the step being a
	// proximal gradient one, at w=0 for stopping condition.
	for (i=0;i<index_length;i++)
		index[i] = i;
	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	f = fun_obj->fun(w0);
	fun_obj->loss_grad(w0, loss_g);
	double alpha0 = fun_obj->vHv(loss_g);
	alpha = min(max(fabs(alpha0), ALPHA_MIN), ALPHA_MAX);

	// initialize with absolute global index
	Q = INF;

	Q0 = Q_prox_grad(step, w0, loss_g, index_length, index, alpha, tmps, &f, &counter);
	if (Q0 == 0)
	{
		info("WARNING: Q0=0\n");
		search = 0;
	}
	delete [] w0;


	f = fun_obj->fun(w);
	fnew = f;
	timer_st = wall_clock_ns();

	while (iter < max_iter && search)
	{
		skip_flag = 0;
		timer_ed = wall_clock_ns();
		accumulated_time += wall_time_diff(timer_ed, timer_st);
		int nnz = 0;
		for (i = 0; i < n; i++)
			nnz += (w[i] != 0);
		if (Q == 0)
		{
			info("WARNING: Update Failed\n");
			if (iter <= 1)
				break;
		}
		else
			info("iter=%d m=%d f=%15.20e Q=%g subprobs=%d w_nnz=%d active_set=%d elapsed_time=%g\n", iter, DynamicM,  f,  Q, counter, nnz, index_length, accumulated_time);
		timer_st = wall_clock_ns();
		
		double stopping = Q / Q0;
		if (iter > 0 && stopping <= eps)
			break;

		fun_obj->loss_grad(w, loss_g);
		
		// Decide if we want to add the new pair of s,y, then update the inner products if added
		// Note that for logistic loss, within any bounded range the D matrix will be of full rank,
		// So the only problem is when we have l < n,
		// but the sparsity-inducing property of the L1 norm will likely shrink n to be smaller than l
		// Thus we almost always have s^T y / s^T s >= sigma > 0

		if (iter != 0)
		{
			daxpy_(&n, &one, loss_g, &inc, tmpy, &inc);
			y0y0 = ddot_(&n, tmpy, &inc, tmpy, &inc);
			if (index_length < n * sparse_factor && index_length > 0)
			{
				s0y0 = 0;
				s0s0 = 0;
				for (int it = 0; it < index_length; it++)
				{
					i = index[it];
					s0y0 += tmpy[i] * tmps[i];
					s0s0 += tmps[i] * tmps[i];
				}
			}
			else
			{
				s0y0 = ddot_(&n, tmpy, &inc, tmps, &inc);
				s0s0 = ddot_(&n, tmps, &inc, tmps, &inc);
			}
			
			if (s0y0 >= update_eps * s0s0)
			{
				memcpy(y + (k*n), tmpy, n * sizeof(double));
				memcpy(s + (k*n), tmps, n * sizeof(double));
				gamma = y0y0 / s0y0;
			}
			else
			{
				info("Skipped updating s and y\n");
				skip_flag = 1;
				skip++;
			}
			DynamicM = min(iter - skip, M);
		}

		memset(step, 0, sizeof(double) * n);
		memset(tmps, 0, sizeof(double) * n);
		if (DynamicM > 0)
		{
			if (!skip_flag)
			{
				if (index_length < n * sparse_factor && index_length > 0)
					update_inner_products(inner_product_matrix, k, DynamicM, s, y, index, index_length);
				else
					update_inner_products(inner_product_matrix, k, DynamicM, s, y);

				compute_R(R, DynamicM, inner_product_matrix, k, gamma);
				k = (k+1)%M;
			}
			alpha = 1;
		}
		else
			alpha = min(max(min(fabs(fun_obj->vHv(loss_g, index, index_length)), gamma), ALPHA_MIN), ALPHA_MAX);
		if (iter > 0)
		{
			if (unchanged_counter >= smooth_trigger)
			{
				if (stage == initial)
					stage = alternating;
			}
			else
				stage = initial;
		}
		if (iter > 0 && !ran_smooth_flag)
		{
			if (nnz == latest_set_size)
				unchanged_counter++;
			else
				unchanged_counter = 0;
			latest_set_size = nnz;
		}

		memcpy(tmpy, loss_g, sizeof(double) * n);
		dscal_(&n, &minus_one, tmpy, &inc);
		if (stage == smooth && ran_smooth_flag)
		{
			ran_smooth_flag = 0;
			alpha = min(max(min(fabs(fun_obj->vHv(loss_g, index, index_length)), gamma), ALPHA_MIN), ALPHA_MAX);
			Q = Q_prox_grad(step, w, loss_g, index_length, index, alpha, tmps, &f, &counter);
			if (Q < 0)
				iter++;
			
			continue;
		}
		else if (iter > 0 && stage != initial && !ran_smooth_flag)
		{
			// all machines conduct the same CG procedure
			std::vector<int> nonzero_set;
			nonzero_set.clear();
			fun_obj->full_grad(w, loss_g, index, index_length, full_g, nonzero_set);
			int nnz_size = (int)nonzero_set.size();
			int max_cg_iter = min(nnz_size, current_max_cg_iter);
			if (stage == alternating)
			{
				max_cg_iter = init_max_cg_iter;
				current_max_cg_iter = init_max_cg_iter;
			}
			double delta = newton(full_g, step, nonzero_set, max_cg_iter, w);
			double step_size;
			if (delta >= 0)
				step_size = 0;
			else
				step_size = fun_obj->smooth_line_search(w, step, delta, eta, nonzero_set, &fnew);

			if (step_size == 1)
			{
				stage = smooth;
				current_max_cg_iter = int(cg_increment * current_max_cg_iter);
			}
			else
			{
				stage = alternating;
				current_max_cg_iter = init_max_cg_iter;
			}
			if (step_size <= 0)
			{
				unchanged_counter = 0;
				ran_smooth_flag = 0;
			}
			else
			{
				info("step_size = %g\n",step_size);
				f = fnew;
				for (i=0;i<nnz_size;i++)
					w[nonzero_set[i]] += step_size * step[i];
				for (i=0;i<nnz_size;i++)
				{
					int idx = nonzero_set[i];
					tmps[idx] = step_size * step[i];
				}

				ran_smooth_flag = 1;
				info("**Smooth Step\n");
				Q = delta;
				counter = 0;
				iter++;
				continue;
			}
		}

		ran_smooth_flag = 0;
		counter = 0;

		if (DynamicM == 0)
		{
			Q = Q_prox_grad(step, w, loss_g, index_length, index, alpha, tmps, &f, &counter);
			if (Q < 0)
				iter++;
		}
		else
		{
			index_length = fun_obj->setselection(w, loss_g, index);
			do
			{
				Q = rpcd(w, loss_g, R, s, y, gamma, step, alpha, DynamicM, index, index_length);
				if (Q == 0)
					break;

				if (index_length < n)
					fnew = fun_obj->f_update(w, step, Q, eta, index, index_length);
				else
					fnew = fun_obj->f_update(w, step, Q, eta);
				alpha *= 2;
				counter++;
			} while (counter < max_modify_Q && (Q > 0 || (fnew - f > eta * Q)));

			if (counter < max_modify_Q && Q < 0)
			{
				f = fnew;
				if (index_length < n)
					for (i=0;i<index_length;i++)
					{
						w[index[i]] += step[i];
						tmps[index[i]] = step[i];
					}
				else
				{
					daxpy_(&n, &one, step, &inc, w, &inc);
					memcpy(tmps, step, sizeof(double) * n);
				}
				iter++;
			}
			else
				Q = 0;
		}
	}

	delete[] step;
	delete[] s;
	delete[] tmpy;
	delete[] tmps;
	delete[] loss_g;
	for (i=0; i < M; i++)
		delete[] inner_product_matrix[i];
	delete[] inner_product_matrix;
	delete[] R;
	delete[] full_g;
	delete[] index;
}

double IPVM_LBFGS::Q_prox_grad(double *step, double *w, double *g, int index_length, int *index, double alpha, double *tmps, double *f, int *counter_ret)
{
	int inc = 1;
	double minus_one = -1.0;
	int counter = 0;
	double *wptr;
	double *gptr;
	const int max_modify_Q = 30;
	int w_size = fun_obj->get_nr_variable();
	int i;
	double Q = 0;
	double fnew = *f;
	if (index_length < w_size)
	{
		wptr = new double[index_length];
		gptr = new double[index_length];
		for (i=0; i<index_length;i++)
		{
			wptr[i] = w[index[i]];
			gptr[i] = g[index[i]];
		}
	}
	else
	{
		wptr = w;
		gptr = g;
	}

	do
	{
		memset(step, 0, sizeof(double) * index_length);
		fun_obj->prox_grad(wptr, gptr, step, step, alpha, index_length);
		Q = fun_obj->regularizer(step, index_length) - fun_obj->regularizer(wptr, index_length);
		daxpy_(&index_length, &minus_one, wptr , &inc, step, &inc);
		Q += 0.5 * alpha * ddot_(&index_length, step, &inc, step, &inc) + ddot_(&index_length, gptr, &inc, step, &inc);
		fnew = fun_obj->f_update(w, step, Q, eta, index, index_length);
		alpha *= 2;
		counter++;
	} while (counter < max_modify_Q && (Q > 0 || (fnew - (*f) > eta * Q)));
	*counter_ret = counter;

	if (counter < max_modify_Q && Q < 0)
	{
		*f = fnew;
		for (i=0;i<index_length;i++)
			w[index[i]] += step[i];
		for (i=0;i<index_length;i++)
		{
			int idx = index[i];
			tmps[idx] = step[i];
		}
	}
	else
		Q = 0;
	if (index_length < w_size)
	{
		delete[] wptr;
		delete[] gptr;
	}
	return Q;
}

void IPVM_LBFGS::update_inner_products(double **inner_product_matrix, int k, int DynamicM, double *s, double *y, int *index, int index_length)
{
	int i;
	int inc = 1;
	char T[] = "T";
	double zero = 0;
	double one = 1.0;
	int n = fun_obj->get_nr_variable();

	double *tmp = new double[DynamicM * 2];
	if (index_length < 0)
	{
		dgemv_(T, &n, &DynamicM, &one, s, &n, s + k * n, &inc, &zero, tmp, &inc);
		dgemv_(T, &n, &DynamicM, &one, y, &n, s + k * n, &inc, &zero, tmp + DynamicM, &inc);
	}
	else
	{
		double *current_s = s + k * n;
		for (i=0;i<DynamicM;i++)
		{
			tmp[i] = 0;
			double *si = s + i * n;
			for (int j=0;j<index_length;j++)
			{
				int it = index[j];
				tmp[i] += current_s[it] * si[it];
			}
		}
		for (i=0;i<DynamicM;i++)
		{
			tmp[DynamicM + i] = 0;
			double *yi = y + i * n;
			for (int j=0;j<index_length;j++)
			{
				int it = index[j];
				tmp[DynamicM + i] += current_s[it] * yi[it];
			}
		}
	}

	for (i=0;i<DynamicM;i++)
	{
		inner_product_matrix[k][2 * i] = tmp[i];
		inner_product_matrix[i][2 * k] = tmp[i];
		inner_product_matrix[k][2 * i + 1] = tmp[DynamicM + i];
	}
	delete[] tmp;
}

void IPVM_LBFGS::compute_R(double *R, int DynamicM, double **inner_product_matrix, int k, double gamma)
{
	int i,j;
	int size = 2 * DynamicM;
	int sizesquare = size * size;

	memset(R, 0, sizeof(double) * sizesquare);

	//R is a symmetric matrix
	//(1,1) block, S^T S
	for (i=0;i<DynamicM;i++)
		for (j=0;j<DynamicM;j++)
			R[i * size + j] = gamma * inner_product_matrix[i][2 * j];

	//(2,2) block, D = diag(s_i^T y_i)
	for (i=0;i<DynamicM;i++)
		R[(DynamicM + i) * (size + 1)] = -inner_product_matrix[i][2 * i + 1];

	//(1,2) block, L = tril(S^T Y, -1), and (2,1) block, L^T
	for (i=1;i<DynamicM;i++)
	{
		int idx = (k + 1 + i) % DynamicM;
		for (j=0;j<i;j++)
		{
			int idxj = (k + 1 + j) % DynamicM;
			R[(DynamicM + idxj) * size + idx] = inner_product_matrix[idx][2 * idxj + 1];
			R[idx * size + DynamicM + idxj] = inner_product_matrix[idx][2 * idxj + 1];
		}
	}

	inverse(R, size);
}

double IPVM_LBFGS::rpcd(double *w, double *loss_g, double *R, double *s, double *y, double gamma, double *step, double scaling, int DynamicM, int *index, int index_length)
{
	const double inner_eps = 1e-2;
	int i,j;
	double dnorm0 = 0;
	double dnorm = 0;
	char N[] = "N";
	double zero = 0.0;
	double one = 1.0;
	int inc = 1;

	int iter = 0;
	int Rsize = 2 * DynamicM;

	double *RUTd = new double[Rsize];
	double *diag = new double[index_length];
	int *perm = new int[index_length];
	double *subsy = new double[index_length * DynamicM * 2];
	double *Rsubsy = new double[index_length * DynamicM * 2];
	double sTs=0, dTd=0;
	double oldd, newd, G;
	double subw;
	int n = fun_obj->get_nr_variable();


	// Reorder to column-major for better memory access
	for (i=0;i<DynamicM;i++)
	{
		double *tmps = s + (n * i);
		double *tmpy = y + (n * i);
		double *subsy1 = subsy + i;
		double *subsy2 = subsy + (DynamicM + i);
		for (j=0;j<index_length;j++)
		{
			subsy1[j * Rsize] = gamma * tmps[index[j]];
			subsy2[j * Rsize] = tmpy[index[j]];
		}
	}
	dgemm_(N, N, &Rsize, &index_length, &Rsize, &one, R, &Rsize, subsy, &Rsize, &zero, Rsubsy, &Rsize);


	memset(step, 0, sizeof(double) * fun_obj->get_nr_variable());
	for (i=0;i<index_length;i++)
		perm[i] = i;
	for (i=0;i<index_length;i++)
		diag[i] = scaling * (gamma - ddot_(&Rsize, subsy + i * Rsize, &inc, Rsubsy + i * Rsize, &inc));

	memset(RUTd, 0, sizeof(double) * Rsize);

	while (iter < max_inner)
	{
		for(i=0; i<index_length; i++)
		{
			int j = i + rand() % (index_length - i);
			swap(perm[i], perm[j]);
		}
		dTd = 0;

		for(i=0; i<index_length; i++)
		{
			int subidx = perm[i];
			int idx = index[subidx];
			oldd = step[subidx];
			G = loss_g[idx] + scaling * (gamma * oldd - ddot_(&Rsize, RUTd, &inc, subsy + subidx * Rsize, &inc));
				
			subw = w[idx];
			fun_obj->prox_grad(&subw, &G, &newd, &oldd, diag[subidx], 1);
			newd -= subw;
			double ddiff = newd - oldd;
			if (fabs(ddiff) > 1e-20)
			{
				step[subidx] = newd;
				dTd += ddiff * ddiff;
				daxpy_(&Rsize, &ddiff, Rsubsy + subidx * Rsize, &inc, RUTd, &inc);
			}
		}
		dnorm = sqrt(dTd);

		if (iter == 0)
			dnorm0 = dnorm;
		if (dnorm < 1e-20 || (iter >= min_inner && dnorm / dnorm0 < inner_eps))
			break;
		iter++;
	}
	info("iters = %d\n",iter);
	delete[] Rsubsy;
	delete[] perm;
	delete[] diag;
	double *tmp = new double[Rsize];

	double obj = 0;
	sTs = 0;
	if (iter > 0)
	{
		for (i=0;i<index_length;i++)
		{
			double oldw = w[index[i]];
			double neww = oldw + step[i];
			obj += fun_obj->regularizer(&neww, 1) - fun_obj->regularizer(&oldw, 1) + loss_g[index[i]] * step[i];
			sTs += step[i] * step[i];
		}
		dgemv_(N, &Rsize, &index_length, &one, subsy, &Rsize, step, &inc, &zero, tmp, &inc);
		obj += scaling * (gamma * sTs - ddot_(&Rsize, RUTd, &inc, tmp, &inc)) / 2;
	}

	delete[] subsy;
	delete[] RUTd;
	delete[] tmp;
	return obj;
}

double IPVM_LBFGS::newton(double *g, double *step, const std::vector<int> &nonzero_set, int max_cg_iter, double *w)
{
	int sub_length = (int) nonzero_set.size();
	double rho = 0.5;
	double c = 1e-6;
	const double psd_threshold = 1e-8;
	double eps_cg = 0.1;

	int i, inc = 1;
	double one = 1;
	double *d = new double[sub_length];
	double *Hd = new double[sub_length];
	double zTr, znewTrnew, alpha, beta, cgtol;
	double *z = new double[sub_length];
	double *r = new double[sub_length];
	double *M = new double[sub_length];
	double sTs = 0, sHs = 0, dHd = 0, dTd = 0;
	double damping = c * pow(dnrm2_(&sub_length, g, &inc),rho);

	double alpha_pcg = 1.0;
	fun_obj->get_diag_preconditioner(M, nonzero_set, w);
	for(i=0; i<sub_length; i++)
		M[i] = (1-alpha_pcg) + alpha_pcg*(M[i] + damping);

	for (i=0; i<sub_length; i++)
	{
		step[i] = 0.0;
		r[i] = -g[i];
		z[i] = r[i] / M[i];
		d[i] = z[i];
	}
	zTr = ddot_(&sub_length, z, &inc, r, &inc);
	double rTr = ddot_(&sub_length, r, &inc, r, &inc);
	double gs = 0.0;

	cgtol = eps_cg * min(1.0, rTr);
	double cg_boundary = 1e+6 * ddot_(&sub_length, z, &inc, z, &inc);
	info("cg_boundary = %g\n",cg_boundary);
	int cg_iter = 0;

	while (cg_iter < max_cg_iter)
	{
		if (sqrt(rTr) <= cgtol)
			break;
		cg_iter++;
		fun_obj->full_Hv(d, Hd, nonzero_set, w);
		daxpy_(&sub_length, &damping, d, &inc, Hd, &inc);
		dHd = ddot_(&sub_length, d, &inc, Hd, &inc);
		dTd = ddot_(&sub_length, d, &inc, d, &inc);
		if (dHd / dTd <= psd_threshold)
		{
			info("WARNING: dHd / dTd <= PSD threshold\n");
			break;
		}

		alpha = zTr/dHd;
		daxpy_(&sub_length, &alpha, d, &inc, step, &inc);
		alpha = -alpha;
		gs = ddot_(&sub_length, g, &inc, step, &inc);
		if (gs >= 0)
		{
			info("gs >= 0 in CG\n");
			daxpy_(&sub_length, &alpha, d, &inc, step, &inc);
			break;
		}

		sTs = ddot_(&sub_length, step, &inc, step, &inc);
		if (sTs >= cg_boundary)
		{
			info("WARNING: reaching cg boundary\n");
			break;
		}

		daxpy_(&sub_length, &alpha, Hd, &inc, r, &inc);
		sHs = -ddot_(&sub_length, r, &inc, step, &inc) - gs;
		if (sHs / sTs <= psd_threshold)
		{
			info("WARNING: sHs / sTs <= PSD threshold\n");
			break;
		}

		for (i=0; i<sub_length; i++)
			z[i] = r[i] / M[i];
		znewTrnew = ddot_(&sub_length, z, &inc, r, &inc);

		beta = znewTrnew/zTr;
		dscal_(&sub_length, &beta, d, &inc);
		daxpy_(&sub_length, &one, z, &inc, d, &inc);
		zTr = znewTrnew;
		rTr= ddot_(&sub_length, r, &inc, r, &inc);
	}
	double delta =  0.5 * sHs + gs;
	if (gs < 0 && delta >= 0)
		delta = gs;

	info("CG Iter = %d\n", cg_iter);

	if (cg_iter == max_cg_iter)
		info("WARNING: reaching maximal number of CG steps\n");

	delete[] d;
	delete[] Hd;
	delete[] z;
	delete[] r;
	delete[] M;

	return delta;
}

IPVM_NEWTON::IPVM_NEWTON(const regularized_fun *fun_obj, double eps, double eta, int max_iter):
	IPVM_LBFGS(fun_obj, eps, 0, eta, max_iter)
{
	this->max_inner = 10;
}

IPVM_NEWTON::~IPVM_NEWTON()
{
}

void IPVM_NEWTON::ipvm_newton(double *w)
{
	const int max_modify_Q = 20;
	const int smooth_trigger = 10;
	const double ALPHA_MAX = 1e10;
	const double ALPHA_MIN = 1e-4;
	const double rho = 0.5;
	const double c = 1e-6;
	//const int switch_threshold = M;
	int inc = 1;
	double one = 1.0;
	double minus_one = -1.0;

	int n = fun_obj->get_nr_variable();

	int i;
	int iter = 0;
	double f;
	double Q0;
	double Q = 0;
	int64_t timer_st, timer_ed;
	double accumulated_time = 0;
	double alpha;

	double *loss_g = new double[n];
	double *step = new double[n];
	double *full_g = new double[n];
	double *tmp = new double[n];
	double fnew;
	int counter = 0;
	int ran_smooth_flag = 0;
	Stage stage = initial;
	int latest_set_size = 0;
	int init_max_cg_iter = 5;
	double cg_increment = 2;
	int current_max_cg_iter = init_max_cg_iter;
	int search = 1;

	// for alternating between previous and current indices
	int *index = new int[n];
	int index_length = n;
	int unchanged_counter = 0;

	// calculate the Q quantity used in update acceptance with the step being a
	// proximal gradient one, at w=0 for stopping condition.
	for (i=0;i<index_length;i++)
		index[i] = i;
	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	f = fun_obj->fun(w0);
	fun_obj->loss_grad(w0, loss_g);
	double alpha0 = fun_obj->vHv(loss_g);
	alpha = min(max(fabs(alpha0), ALPHA_MIN), ALPHA_MAX);

	// initialize with absolute global index
	Q = INF;

	Q0 = Q_prox_grad(step, w0, loss_g, index_length, index, alpha, tmp, &f, &counter);
	if (Q0 == 0)
	{
		info("WARNING: Q0=0\n");
		search = 0;
	}
	delete [] w0;


	f = fun_obj->fun(w);
	fnew = f;
	timer_st = wall_clock_ns();

	while (iter < max_iter && search)
	{
		timer_ed = wall_clock_ns();
		accumulated_time += wall_time_diff(timer_ed, timer_st);
		int nnz = 0;
		for (i = 0; i < n; i++)
			nnz += (w[i] != 0);
		if (Q == 0)
		{
			info("WARNING: Update Failed\n");
			if (iter <= 1)
				break;
		}
		else
			info("iter=%d f=%15.20e Q=%g subprobs=%d w_nnz=%d active_set=%d elapsed_time=%g\n", iter, f,  Q, counter, nnz, index_length, accumulated_time);
		timer_st = wall_clock_ns();
		
		double stopping = Q / Q0;
		if (iter > 0 && stopping <= eps)
			break;

		fun_obj->loss_grad(w, loss_g);
		
		memset(step, 0, sizeof(double) * n);
		alpha = 1;
		if (iter > 0)
		{
			if (unchanged_counter >= smooth_trigger)
			{
				if (stage == initial)
					stage = alternating;
			}
			else
				stage = initial;
		}
		if (iter > 0 && !ran_smooth_flag)
		{
			if (nnz == latest_set_size)
				unchanged_counter++;
			else
				unchanged_counter = 0;
			latest_set_size = nnz;
		}

		if (stage == smooth && ran_smooth_flag)
		{
			ran_smooth_flag = 0;
			alpha = min(max(fabs(fun_obj->vHv(loss_g, index, index_length)), ALPHA_MIN), ALPHA_MAX);
			Q = Q_prox_grad(step, w, loss_g, index_length, index, alpha, tmp, &f, &counter);
			if (Q < 0)
				iter++;
			
			continue;
		}
		else if (iter > 0 && stage != initial && !ran_smooth_flag)
		{
			// all machines conduct the same CG procedure
			std::vector<int> nonzero_set;
			nonzero_set.clear();
			fun_obj->full_grad(w, loss_g, index, index_length, full_g, nonzero_set);
			int nnz_size = (int)nonzero_set.size();
			int max_cg_iter = min(nnz_size, current_max_cg_iter);
			if (stage == alternating)
			{
				max_cg_iter = init_max_cg_iter;
				current_max_cg_iter = init_max_cg_iter;
			}
			double delta = newton(full_g, step, nonzero_set, max_cg_iter, w);
			double step_size;
			if (delta >= 0)
				step_size = 0;
			else
				step_size = fun_obj->smooth_line_search(w, step, delta, eta, nonzero_set, &fnew);

			if (step_size == 1)
			{
				stage = smooth;
				current_max_cg_iter = int(cg_increment * current_max_cg_iter);
			}
			else
			{
				stage = alternating;
				current_max_cg_iter = init_max_cg_iter;
			}
			if (step_size <= 0)
			{
				unchanged_counter = 0;
				ran_smooth_flag = 0;
			}
			else
			{
				info("step_size = %g\n",step_size);
				f = fnew;
				for (i=0;i<nnz_size;i++)
					w[nonzero_set[i]] += step_size * step[i];

				ran_smooth_flag = 1;
				info("**Smooth Step\n");
				Q = delta;
				counter = 0;
				iter++;
				continue;
			}
		}

		ran_smooth_flag = 0;
		counter = 0;

		memset(step, 0, sizeof(double) * n);
		fun_obj->prox_grad(w, loss_g, step, step, one);
		daxpy_(&n, &minus_one, w, &inc, step, &inc);
		double damping = c * pow(dnrm2_(&n, step, &inc), rho);

		index_length = fun_obj->setselection(w, loss_g, index);
		do
		{
			Q = rpcd(w, loss_g, step, damping, alpha, index, index_length);

			if (Q == 0)
				break;
			if (index_length < n)
				fnew = fun_obj->f_update(w, step, Q, eta, index, index_length);
			else
				fnew = fun_obj->f_update(w, step, Q, eta);
			alpha *= 2;
			counter++;
		} while (counter < max_modify_Q && (Q > 0 || (fnew - f > eta * Q)));

		if (counter < max_modify_Q && Q < 0)
		{
			f = fnew;
			if (index_length < n)
				for (i=0;i<index_length;i++)
					w[index[i]] += step[i];
			else
				daxpy_(&n, &one, step, &inc, w, &inc);
			iter++;
		}
		else
			Q = 0;
	}

	delete[] step;
	delete[] loss_g;
	delete[] full_g;
	delete[] index;
	delete[] tmp;
}

double IPVM_NEWTON::rpcd(double *w, double *loss_g, double *step, double damping, double scaling, int *index, int index_length)
{
	const double inner_eps = 1e-2;
	int i;
	double dnorm0 = 0;
	double dnorm;
	double *diag = new double[index_length];
	int *perm = new int[index_length];
	int intermediate_vec_size = fun_obj->get_intermediate_size();
	double *intermediate = new double[intermediate_vec_size];

	int iter = 0;

	double oldd, newd, G;
	double dTd = 0, sTs = 0;
	double subw;

	memset(step, 0, sizeof(double) * fun_obj->get_nr_variable());
	memset(intermediate, 0, sizeof(double) * intermediate_vec_size);
	fun_obj->diagH(diag, index, index_length);
	for (i=0;i<index_length;i++)
	{
		perm[i] = i;
		diag[i] = scaling * (diag[i] + damping);
	}

	while (iter < max_inner)
	{
		for(i=0; i<index_length; i++)
		{
			int j = i + rand() % (index_length - i);
			swap(perm[i], perm[j]);
		}
		dTd = 0;
		sTs = 0;

		for(i=0; i<index_length; i++)
		{
			int subidx = perm[i];
			int idx = index[subidx];
			oldd = step[subidx];
			G = loss_g[idx] + scaling * (damping * oldd + fun_obj->subHv(intermediate, idx));
			subw = w[idx];
			fun_obj->prox_grad(&subw, &G, &newd, &oldd, diag[subidx], 1);
			newd -= subw;
			sTs += newd * newd;
			double ddiff = newd - oldd;
			if (fabs(ddiff) > 1e-20)
			{
				step[subidx] = newd;
				dTd += ddiff * ddiff;
				fun_obj->update_intermediate(intermediate, ddiff, idx);
			}
		}
		dnorm = sqrt(dTd);
		if (iter == 0)
			dnorm0 = dnorm;
		if (dnorm < 1e-20 || (iter >= min_inner && dnorm / dnorm0 < inner_eps))
			break;
		iter++;
	}

	info("iters = %d\n",iter);

	double obj = 0;
	if (iter > 0)
	{
		for (i=0;i<index_length;i++)
		{
			double oldw = w[index[i]];
			double neww = oldw + step[i];
			obj += fun_obj->regularizer(&neww, 1) - fun_obj->regularizer(&oldw, 1) + loss_g[index[i]] * step[i];
		}
		obj += scaling * (damping * sTs + fun_obj->vHv_from_intermediate(intermediate)) / 2;
	}
	delete[] diag;
	delete[] intermediate;
	delete[] perm;
	return obj;
}

// end of solvers

static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w)
{
	double eps = param->eps;

	int l = prob->l;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = l - pos;

	double primal_solver_tol = (eps*max(min(pos,neg), 1))/l;
	if (param->problem_type == LASSO)
		primal_solver_tol = eps;

	l1r_fun *l1r_fun_obj=NULL;
	problem prob_col;
	feature_node *x_space = NULL;
	transpose(prob, &x_space ,&prob_col);

	switch(param->problem_type)
	{
		case L1R_LR:
			l1r_fun_obj = new l1r_lr_fun(&prob_col, param->C);
			break;
		case LASSO:
			l1r_fun_obj = new lasso(&prob_col, param->C);
			break;
		case L1R_L2_LOSS_SVC:
			l1r_fun_obj = new l1r_l2_svc_fun(&prob_col, param->C);
			break;
		default:
			fprintf(stderr, "ERROR: unknown problem_type\n");
	}

	switch(param->solver_type)
	{
		case SOLVER_LBFGS:
		{
			IPVM_LBFGS ipvm_lbfgs_obj(l1r_fun_obj, primal_solver_tol, param->m, param->eta);
			ipvm_lbfgs_obj.set_print_string(liblinear_print_string);
			ipvm_lbfgs_obj.ipvm_lbfgs(w);
			break;
		}
		case SOLVER_NEWTON:
		{
			IPVM_NEWTON ipvm_newton_obj(l1r_fun_obj, primal_solver_tol, param->eta);
			ipvm_newton_obj.set_print_string(liblinear_print_string);
			ipvm_newton_obj.ipvm_newton(w);
			break;
		}
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
	}


	delete l1r_fun_obj;
	delete [] prob_col.y;
	delete [] prob_col.x;
	delete [] x_space;
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		model_->w = Malloc(double, w_size);
		for(i=0; i<w_size; i++)
			model_->w[i] = 0;
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, model_->w);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		if(nr_class == 2)
		{
			model_->w=Malloc(double, w_size);

			int e0 = start[0]+count[0];
			k=0;
			for(; k<e0; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			for(i=0;i<w_size;i++)
				model_->w[i] = 0;

			train_one(&sub_prob, param, model_->w);
		}
		else
		{
			model_->w=Malloc(double, w_size*nr_class);
			double *w=Malloc(double, w_size);
			for(i=0;i<nr_class;i++)
			{
				int si = start[i];
				int ei = si+count[i];

				k=0;
				for(; k<si; k++)
					sub_prob.y[k] = -1;
				for(; k<ei; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				for(j=0;j<w_size;j++)
					w[j] = 0;

				train_one(&sub_prob, param, w);

				for(j=0;j<w_size;j++)
					model_->w[j*nr_class+i] = w[j];
			}
			free(w);
		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
	}
	return model_;
}


double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(check_regression_model(model_))
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *problem_type_table[]=
{
	"L1R_LR", "LASSO","L1R_L2_LOSS_SVC",NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "problem_type %s\n", problem_type_table[param.problem_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.17g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.17g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);


	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	free(model_->label);\
	free(model_);\
	return NULL;\
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"problem_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;problem_type_table[i];i++)
			{
				if(strcmp(problem_type_table[i],cmd)==0)
				{
					param.problem_type=i;
					break;
				}
			}
			if(problem_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
	}


	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(label_idx < 0 || label_idx >= nr_class)
		return 0;
	if(nr_class == 2)
		if(label_idx == 0)
			return w[idx];
		else
			return -w[idx];
	else
		return w[idx*nr_class+label_idx];
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != SOLVER_LBFGS
	&& param->solver_type != SOLVER_NEWTON)
		return "unknown solver type";

	if(param->problem_type != L1R_LR
	&& param->problem_type != LASSO
	&& param->problem_type != L1R_L2_LOSS_SVC)
		return "unknown problem type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.problem_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.problem_type==LASSO);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}
