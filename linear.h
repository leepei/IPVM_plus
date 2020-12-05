#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

extern int liblinear_version;

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */
	int global_l;
};

/* solver_type */
enum {SOLVER_LBFGS, SOLVER_NEWTON};
enum {L1R_LR, LASSO, L1R_L2_LOSS_SVC};

struct parameter
{
	int solver_type;
	int problem_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int m;
	double eta;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
double get_decfun_bias(const struct model *model_, int label_idx);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
int check_regression_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

void fill_range(std::vector<int> &v, int n);
