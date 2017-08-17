// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 or the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2016
 * Dep. Of Computer Science
 * Technical University of Munich (TUM)
 *
 */

#include "stream_based_al_forest.h"

/*
 * Mondrian block
 */
MondrianBlock::MondrianBlock(const int& feature_dim, 
        const mondrian_settings& settings) :
    feature_dim_(feature_dim),
    /* Initialize block borders with zeros */
    min_block_dim_(feature_dim, arma::fill::zeros),
    max_block_dim_(feature_dim, arma::fill::zeros),
    debug_(settings.debug) {

    min_block_dim_.fill(numeric_limits<float>::infinity());
    max_block_dim_.fill(-numeric_limits<float>::infinity());
    if (debug_)
        cout << "### Init Mondrian Block 1" << endl;
}

MondrianBlock::MondrianBlock(const int& feature_dim,
        arma::fvec& min_block_dim, arma::fvec& max_block_dim,
        const mondrian_settings& settings) :
    feature_dim_(feature_dim), 
    min_block_dim_(min_block_dim),
    max_block_dim_(max_block_dim),
    debug_(settings.debug) {

    update_sum_dim_range();

    if (debug_)
        cout << "### Init Mondrian Block 2" << endl;
}

MondrianBlock::~MondrianBlock() {
    /* Delete all pointer elements */
}

/**
 * Get dimension range
 */
float MondrianBlock::get_dim_range(const arma::fvec& cur_sample) {

    arma::fvec min_block_dim = arma::min(min_block_dim_, cur_sample);
    arma::fvec max_block_dim = arma::max(max_block_dim_, cur_sample);
    float sum_dim_range =  arma::accu(max_block_dim - min_block_dim);
    return sum_dim_range;
}
/*
* Calculate sum of all dimension = sum( max_block_dim_ - min_block_dim_ )
*/
void MondrianBlock::update_sum_dim_range() {
    /* accu: same as to get the sum, regardless of the argument type */
    sum_dim_range_ = arma::accu(max_block_dim_ - min_block_dim_);
}

/*
* Updates minimum and maximum of training data at this block
*/
void MondrianBlock::update_range_states(const arma::fvec& cur_min_dim, 
        const arma::fvec& cur_max_dim) {
    if (debug_)
        cout << "### [MondrianBlock] - update_range_states" << endl;
    if (int(min_block_dim_.size()) == feature_dim_ && 
            int(max_block_dim_.size()) == feature_dim_) {
        min_block_dim_ = arma::min(min_block_dim_, cur_min_dim);
        max_block_dim_ = arma::max(max_block_dim_, cur_max_dim);
    } else {
        /* Boundaries are not initialized yet */
        min_block_dim_ = cur_min_dim;
        max_block_dim_ = cur_max_dim;
    }

    if (debug_) {
        cout << "min: " << min_block_dim_ << endl;
        cout << "max: " << max_block_dim_ << endl;
    }
    update_sum_dim_range();
}

/*
 * Update minimum and maximum of training data at this block
 */
void MondrianBlock::update_range_states(const arma::fvec& cur_point) {
    if (debug_)
        cout << "### [MondrianBlock] - update_range_states" << endl;
    if (int(min_block_dim_.size()) == feature_dim_ && 
            int(max_block_dim_.size()) == feature_dim_) {
        min_block_dim_ = arma::min(min_block_dim_, cur_point);
        max_block_dim_ = arma::max(max_block_dim_, cur_point); 
    } else {
        /* Boundaries are not initialized yet */
        min_block_dim_ = cur_point;
        max_block_dim_ = cur_point;
    }
    update_sum_dim_range();
}

/*
 * Serialization of Mondrian block
 */
std::ostream & operator<<(std::ostream &os, const MondrianBlock &mb) {
    /*
     * return os << mb.feature_dim_ << ' '<< mb.sum_dim_range_ << ' '
     *     << mb.min_block_dim_ << ' '<< mb.max_block_dim_ << ' ' <<
     *     mb.debug_ << ' ';
     */
    return os << mb.feature_dim_ << mb.sum_dim_range_ <<
        mb.min_block_dim_ << mb.max_block_dim_
        << mb.debug_;
}

/*---------------------------------------------------------------------------*/
/* Initialize random generator */
RandomGenerator MondrianNode::random;

/*
 * Construct tree node
 */
MondrianNode::MondrianNode(int* num_classes, const int& feature_dim, 
       const float& budget, MondrianNode& parent_node,
       const mondrian_settings& settings, int& depth,
       float& decision_distr_param_alpha,
       float& decision_distr_param_beta) :
    num_classes_(num_classes),
    data_counter_(0),
    is_leaf_(true),
    split_dim_(0),
    split_loc_(0.),
    max_split_costs_(budget),
    budget_(budget),
    settings_(&settings),
    depth_(depth),
    decision_distr_param_alpha_(decision_distr_param_alpha),
    decision_distr_param_beta_(decision_distr_param_beta){
    
    if (settings_->debug)
        cout << "### Init Mondrian Node 1 " << this << endl;
    /* Initialize Mondrian block */
    mondrian_block_ = new MondrianBlock(feature_dim, settings);   
    id_parent_node_ = &parent_node; /* Pass parent node */
    id_left_child_node_ = NULL;
    id_right_child_node_ = NULL;
    pred_prob_ = arma::fvec(*num_classes, arma::fill::zeros);
}

/*
 * Construct tree node with given values of boundaries of
 * Mondrian block
 */
MondrianNode::MondrianNode(int* num_classes, const int& feature_dim, 
        const float& budget, MondrianNode& parent_node, 
        arma::fvec& min_block_dim, arma::fvec& max_block_dim,
        const mondrian_settings& settings, int& depth,
        float& decision_distr_param_alpha,
        float& decision_distr_param_beta) :
    num_classes_(num_classes),
    data_counter_(0),
    is_leaf_(true),  /* Gets false if function set_child_node is called */
    split_dim_(0),
    split_loc_(0.),
    max_split_costs_(budget),
    budget_(budget),
    settings_(&settings),
    depth_(depth),
    decision_distr_param_alpha_(decision_distr_param_alpha),
    decision_distr_param_beta_(decision_distr_param_beta){
    
    if (settings_->debug)
        cout << "### Init Mondrian Node 2 " << this << endl;
    /* Initialize Mondrian block */
    mondrian_block_ = new MondrianBlock(feature_dim, min_block_dim, 
            max_block_dim, settings);
    id_parent_node_ = &parent_node;  /* Pass parent node */
    id_left_child_node_ = NULL;
    id_right_child_node_ = NULL;
    pred_prob_ = arma::fvec(*num_classes, arma::fill::zeros);
}

/*
 * Construct tree node with given values of boundaries of
 * the Mondrian block and one existing child node
 */
MondrianNode::MondrianNode(int* num_classes, const int& feature_dim, 
        const float& budget, MondrianNode& parent_node, 
        MondrianNode& left_child_node, MondrianNode& right_child_node, 
        arma::fvec& min_block_dim, arma::fvec& max_block_dim,
        const mondrian_settings& settings, int& depth,
        float& decision_distr_param_alpha,
        float& decision_distr_param_beta) :
    num_classes_(num_classes),
    data_counter_(0),
    is_leaf_(false),
    split_dim_(0),
    split_loc_(0.),
    max_split_costs_(budget),
    budget_(budget),
    settings_(&settings),
    depth_(depth),
    decision_distr_param_alpha_(decision_distr_param_alpha),
    decision_distr_param_beta_(decision_distr_param_beta) {

    if (settings_->debug)
        cout << "### Init Mondrian Node 3 " << this << endl;
    /* Initialize Mondrian block */
    mondrian_block_ = new MondrianBlock(feature_dim, min_block_dim,
            max_block_dim, settings);
    /* Pass parent and child nodes */
    id_parent_node_ = &parent_node;
    id_left_child_node_ = &left_child_node;
    id_right_child_node_ = &right_child_node;
    pred_prob_ = arma::fvec(*num_classes, arma::fill::zeros);
}

MondrianNode::~MondrianNode() {
    /* Delete all pointer elements */
    if (!is_leaf_) {
        delete id_left_child_node_;
        id_left_child_node_ = NULL;
        delete id_right_child_node_;
        id_left_child_node_ = NULL;
    }
    /* Delete mondrian block */
    delete mondrian_block_;
    mondrian_block_ = NULL;
}

/*
 * Print information of current node
 */
void MondrianNode::print_info() {
    cout << endl;
    cout << "-------------------" << endl;
    cout << "node: " << this << endl;
    cout << "split_dim:         " << split_dim_ << endl;
    cout << "split_loc:         " << split_loc_ << endl;
    cout << "max_split_cost:    " << max_split_costs_ << endl;
    cout << "budget:            " << budget_ << endl; 
    cout << "data_counter:      " << data_counter_ << endl;
    cout << "class histogram:   " << endl;
    cout << count_labels_ << endl;
    
    if (mondrian_block_->get_feature_dim() < 10) {
        cout << "---------------" << endl;
        cout << "block:" << endl;
        cout << "min_block: " << endl;
        cout << mondrian_block_->get_min_block_dim() << endl;
        cout << "max_block: " << endl;
        cout << mondrian_block_->get_max_block_dim() << endl;
    }

    cout << "---------------" << endl;
    cout << "parent:        " << id_parent_node_ << endl;
    cout << "left child:    " << id_left_child_node_ << endl;
    cout << "right child:   " << id_right_child_node_ << endl;
    cout << endl;
    if (id_left_child_node_ != NULL)
        id_left_child_node_->print_info();
    if (id_right_child_node_ != NULL) 
        id_right_child_node_->print_info();

}

/*
 * Update histogram with additional class and increase class histogram +1
 */
void MondrianNode::add_new_class() {
    if (settings_->debug)
        cout << "### add_new_class" << endl;
    /* Size of count_labels should be one smaller than num_classes */
    assert(*num_classes_ > int(count_labels_.size())); 
    if (*num_classes_ >= int(count_labels_.size())) {
        //count_labels_.push_back(0);
        //count_labels_.resize(count_labels_.size()+1); 
        count_labels_.resize(*num_classes_); 
        pred_prob_.resize(*num_classes_); 
    }
    /* Update child nodes */
    if (id_left_child_node_ != NULL) {
        id_left_child_node_->add_new_class();
    }
    if (id_right_child_node_ != NULL) {
        id_right_child_node_->add_new_class();
    }
}

/*
 * Predict class of current sample
 */
int MondrianNode::predict_class(Sample& sample, arma::fvec& pred_prob, 
        float& prob_not_separated_yet, mondrian_confidence& m_conf) {

    if (settings_->debug)
        cout << "predict_class..." << endl;
    int pred_class = -1;
    /* 
     * If x lies outside B^x_j at node j, the probability that x will branch 
     * off into its own node at node j, denoted by p^s_j(x), is equal to the
     * probability that a split exists in B_j outside B^x_j 
     */
    int feature_dimension = mondrian_block_->get_feature_dim();
    arma::fvec zero_vec(feature_dimension, arma::fill::zeros);
    /* \eta_j(x) */
    float expo_param = 1.0;
    expo_param = arma::accu(arma::max(zero_vec, 
                (sample.x - mondrian_block_->get_max_block_dim()))) + 
        arma::accu(arma::max(zero_vec, 
                    (mondrian_block_->get_min_block_dim() - sample.x)));
    /* Compute mondrian confidence values */
    if (is_leaf_) {
        /* 1. Compute euclidean distance */
        m_conf.distance = arma::norm(arma::max(zero_vec, 
                    (sample.x - mondrian_block_->get_max_block_dim())),2) + 
            arma::norm(arma::max(zero_vec, 
                        (mondrian_block_->get_min_block_dim() - sample.x)),2);
        /* 2. Get number of samples at current node */
        m_conf.number_of_points = arma::accu(id_parent_node_->count_labels_);
        /* 3. Calculate density of current mondrian block */
        //arma::fvec tmp_vec = id_parent_node_->mondrian_block_->get_max_block_dim() - 
        //   id_parent_node_->mondrian_block_->get_min_block_dim();
        //arma::fvec tmp_vec = mondrian_block_->get_max_block_dim() - mondrian_block_->get_min_block_dim();
        m_conf.density = expo_param;
    }
    /* Probability that x_i will branch off into its own node at node j */
    float prob_not_separated_now = exp(-expo_param * max_split_costs_);
    float prob_separated_now = 1 - prob_not_separated_now;  /* p^s_j(x) */ 
    if (settings_->debug) {
        cout << "prob_not_separated_now: " << prob_not_separated_now << endl;
        cout << "prob_separated_now: " << prob_separated_now << endl;
    }
    arma::fvec base = get_prior_mean();
    
    float discount = exp(-settings_->discount_param * max_split_costs_);
    
    if (settings_->debug)
        cout << "discount: " << discount << endl;
    /* Interpolated Kneser Ney smoothing */
    arma::Col<arma::uword> cnt(*num_classes_, arma::fill::zeros);
    if (is_leaf_) {
        cnt = count_labels_;
    } else {
        arma::Col<arma::uword> ones_vec(*num_classes_, arma::fill::ones);
        cnt = arma::min(count_labels_, ones_vec);
    }

    /* Check if current sample lies outside */
    // or expo_param > 0
    if (greater_zero(expo_param)) {
        /* 
         * Compute expected discount d, where \delta is drawn from a truncated
         * exponential with rate \eta_j(x), truncated to the interval
         * [0, \delta]
         */
        arma::fvec cnt_f = arma::conv_to<arma::fvec>::from(cnt);
        arma::fvec ones_vec(cnt_f.size(),arma::fill::ones);
        arma::fvec num_tables_k = arma::min(cnt_f, ones_vec);
        float num_customers = float(arma::sum(cnt));
        float num_tables = float(arma::sum(num_tables_k));
        
        /* 
         * Expected discount is averaging over time of cut which is
         * a truncated exponential
         */
        discount = (expo_param / (expo_param + settings_->discount_param)) *
            (-(exp(-(expo_param + settings_->discount_param) *
             max_split_costs_) - 1)) / 
            (-(exp(-expo_param * max_split_costs_)-1));

        float discount_per_num_customers = discount / num_customers;
        arma::fvec pred_prob_tmp = (num_tables * discount_per_num_customers *
            base) + (cnt_f / num_customers) - (discount_per_num_customers * 
            num_tables_k);

        pred_prob += prob_separated_now * prob_not_separated_yet * pred_prob_tmp;
        prob_not_separated_yet *= prob_not_separated_now;
    }
    /* c_j,k: number of customers at restaurant j eating dish k */     
    /* Compute posterior mean normalized stable */
    if (!is_leaf_) {
        if (equal(sample.x[split_dim_],split_loc_) || sample.x[split_dim_]
                < split_loc_) {

            if (settings_->debug) 
                cout << "left" << endl;
            pred_class = id_left_child_node_->predict_class(sample, pred_prob,
                    prob_not_separated_yet, m_conf);
        } else {
            if (settings_->debug) 
                cout << "right" << endl;
            pred_class = id_right_child_node_->predict_class(sample, pred_prob,
                    prob_not_separated_yet, m_conf);
        }
    } else if (is_leaf_ && greater_zero(expo_param) == false) {
        pred_prob = compute_posterior_mean_normalized_stable(
                cnt, discount, base) * prob_not_separated_yet;
    }
    /* Get class with highest probability */
    /* Check if all classes have same probability -> return -2 */
    if (equal_elements(pred_prob))
        return -2;
    float tmp_value = 0.;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > tmp_value) {
            tmp_value = pred_prob[i];
            pred_class = i;
        }
    }
    return pred_class;
}

/*
 * Return address of root node
 */
MondrianNode* MondrianNode::update_root_node() {
    
    if (id_parent_node_ != NULL) {
        return id_parent_node_->update_root_node();
    } else {
        return this;
    }
}

/*
 * Update current data sample
 */
void MondrianNode::update(const Sample& sample) {
    /*
     * Additional check for the case that less than
     * two data points passed by at the root node
     */
    if (id_parent_node_ == NULL && data_counter_ < 1) {
        mondrian_block_->update_range_states(sample.x);
        sample_mondrian_block(sample); /* Set max_split_cost */
        add_training_point_to_node(sample);
    } else { 
        extend_mondrian_block(sample);
    }
}

/*
 * Checks if all labels in a node are identical
 * - go through vector count_labels_ and check if only one element is > 1
 */
bool MondrianNode::check_if_same_labels() {
    if (settings_->debug) 
        cout << "### pause_mondrian()" << endl;
    bool same_labels = false;
    /* Function "count" returns number of values that are zero */
    arma::Col<arma::uword> zero_elem = arma::find(count_labels_ < 1);
    if (zero_elem.size() == ((unsigned int)count_labels_.size()-1) ||
            count_labels_.size() <= 1) {
        same_labels = true;
    }
    if (settings_->max_samples_in_one_node > 0) {
      if (data_counter_ > settings_->max_samples_in_one_node) {
        same_labels = false;
      }
    }

    return same_labels;
}

/*
 * Checks if all labels and the current point in a node are identical
 */
bool MondrianNode::check_if_same_labels(const Sample& sample){
    if (settings_->debug) 
        cout << "### check_same_labels(sample)" << endl;
    bool same_labels = false;
    /* Function "count" returns number of values that are zero */
    arma::Col<arma::uword> zero_elem = arma::find(count_labels_ < 1);
    unsigned int count_val = zero_elem.size();
    if (count_val == (unsigned int)count_labels_.size()) {
        /* All elements are zero */
        same_labels = true;
    } else if (count_val == ((unsigned int)count_labels_.size()-1)) {
        same_labels = true; /* Is true if only one value is greater than 0 */
        /* 
        * Check if the only label of the current node has the same label
        * as the label of the current sample
        */
        if (count_labels_.size() > 1) {
            if (count_labels_[sample.y] > 0) {
                same_labels = true;
            } else {
                same_labels = false;
            }
        }
    }

    if (settings_->debug) 
        cout << "### " << same_labels << endl;
    return same_labels;
}

/*
 * Checks if a Mondrian block should be paused
 * - pause if all labels in a node are identical
 */
bool MondrianNode::pause_mondrian() {
    return check_if_same_labels();
}

/*
 * Update statistic (posterior) of current node
 */
void MondrianNode::update_posterior_node_incremental(const Sample& sample) {
    /* 
     * Size of count_labels_ has to be greater than current value of
     * sample.y -> Else: current element of count_labels does not exist
     */
    ++data_counter_;
    assert(int(count_labels_.size()) > sample.y);
    count_labels_[sample.y] += 1; 
}

/*
 * Initialize update posterior node
 */
void MondrianNode::init_update_posterior_node_incremental(
        MondrianNode& node_id, const Sample& sample) {
    if (&node_id == NULL) { 
        
        /* 
         * Initialize histogram of current node with zeros.
         * Size of histogram depends on current number of classes.
         */
        count_labels_ = arma::Col<arma::uword>(*num_classes_,
                arma::fill::zeros);
        data_counter_ = 0;
    } else {
        /* Copy histogram of node "node_id" */
        count_labels_ = node_id.count_labels_;
        /* Update data counter */
        data_counter_ = node_id.data_counter_;
    }
    update_posterior_node_incremental(sample);
}

/*
 * Initialize update posterior node (copy histogram of parent node)
 */
void MondrianNode::init_update_posterior_node_incremental(
        MondrianNode& node_id) {
    if (&node_id == NULL) { 
        /* 
         * Initialize histogram of current node with zeros.
         * Size of histogram depends on current number of classes.
         */
        count_labels_ = arma::Col<arma::uword>(*num_classes_,
                arma::fill::zeros);
        data_counter_ = 0;
    } else {
        count_labels_ = node_id.count_labels_;
        data_counter_ = node_id.data_counter_;
    }
}

/*
 * Add a training data point to current node
 */
void MondrianNode::add_training_point_to_node(const Sample& sample) {
    update_posterior_node_incremental(sample);    
}

/*
 * Compute new boundary values of new Mondrian block, which
 * depends on the split dimension and split location
 */
std::pair<arma::fvec, arma::fvec> 
MondrianNode::compute_left_right_statistics(
        int& split_dim, float& split_loc, const arma::fvec& sample_x,
        arma::fvec min_cur_block, arma::fvec max_cur_block,
        bool left_split) {
    std::vector<arma::fvec> points;

    if (left_split) {
        if (sample_x[split_dim] <= split_loc) {
            points.push_back(sample_x);
        }         
        if (min_cur_block[split_dim] <= split_loc) {
            points.push_back(min_cur_block);
        }         
        if (max_cur_block[split_dim] <= split_loc) {
            points.push_back(max_cur_block);
        }     
    } else {
        if (sample_x[split_dim] > split_loc) {
            points.push_back(sample_x);
        }         
        if (min_cur_block[split_dim] > split_loc) {
            points.push_back(min_cur_block);
        }         
        if (max_cur_block[split_dim] > split_loc) {
            points.push_back(max_cur_block);
        }     
    }
    /* Calculate min and max boundary values */
    std::vector<arma::fvec>::iterator it = points.begin();
    arma::fvec tmp_min(sample_x.size());
    arma::fvec tmp_max(sample_x.size());
    if (it != points.end()) {
        tmp_min = *it;
        tmp_max = *it;
    }
    for ( ; it < points.end(); it++) {
       tmp_min = arma::min(tmp_min,*it); 
       tmp_max = arma::max(tmp_max,*it); 
    }

    std::pair<arma::fvec, arma::fvec> new_block(tmp_min, tmp_max);
    return new_block; 
}

/*
 * Pass child node and set variable "is_leaf_" to false
 */
void MondrianNode::set_child_node(MondrianNode& child_node, 
        bool is_left_node) {

    if (is_left_node) {
        id_left_child_node_ = &child_node; 
    } else {
        id_right_child_node_ = &child_node;
    }
    /* Current node has a child, therefore, it is not longer a leaf node */
    is_leaf_ = false;
}

/**
 * Get counts to calculate posterior inference (Chinese restaurant)
 */
void MondrianNode::get_counts(int& num_tables_k, int& num_customers, 
        int& num_tables) {
    
}

/*
 * Compute prior mean
 */
arma::fvec MondrianNode::get_prior_mean(arma::fvec& pred_prob_par) {
    arma::fvec base(*num_classes_,arma::fill::ones);
    if (id_parent_node_ == NULL) {
        base = base / *num_classes_;
    } else {
        base = pred_prob_par;
    }
    return base;
}
arma::fvec MondrianNode::get_prior_mean() {
    arma::fvec base(*num_classes_,arma::fill::ones);
    if (id_parent_node_ == NULL) {
        base = base / *num_classes_;
    } else {
        base = id_parent_node_->pred_prob_;
    }
    return base;
}
/*
 * Compute posterior mean
 */
arma::fvec MondrianNode::compute_posterior_mean_normalized_stable(
        arma::Col<arma::uword>& cnt, float& discount,
        arma::fvec& base) {
    if (settings_->debug)
        cout << "compute_posterior....." << endl;
    arma::fvec cnt_f = arma::conv_to<arma::fvec>::from(cnt);
    arma::fvec ones_vec(cnt_f.size(),arma::fill::ones);
    arma::fvec num_tables_k = arma::min(cnt_f, ones_vec);
    float num_customers = float(arma::sum(cnt));
    float num_tables = float(arma::sum(num_tables_k));
    /* Calculate probability of each class */
    arma::fvec pred_prob = (cnt_f - discount * num_tables_k + 
            discount * num_tables * base) / num_customers;

    return pred_prob;
}

void MondrianNode::update_depth() {
    depth_++;
    if (id_left_child_node_ != NULL)
        id_left_child_node_->update_depth();
    if (id_right_child_node_ != NULL)
        id_right_child_node_->update_depth();
}
/*
 * Update split cost, split dimension, split location
 */
void MondrianNode::sample_mondrian_block(const Sample& sample, 
        bool create_new_leaf) {

    if (settings_->debug)
        cout << "### sample_mondrian_block-----------------" << endl;
    bool if_paused = check_if_same_labels(sample);

    float split_cost = 0.0;
    float dim_range = 1.0;
    dim_range = mondrian_block_->get_sum_dim_range();
    if (equal(dim_range, 0.0)) {
        dim_range = mondrian_block_->get_dim_range(sample.x);
        if (greater_zero(dim_range)) {
            create_new_leaf = true;
        } else {
            dim_range = 0.0;
        }
    } else if (dim_range < -1) {
        dim_range = 0.0;
    }
    if (if_paused == true || equal(dim_range,0.)) { 
        split_cost = numeric_limits<float>::infinity();
        max_split_costs_ = budget_;
    } else {
        split_cost = random.rand_exp_distribution(dim_range);
        max_split_costs_ = split_cost;
    }
    float new_budget = budget_ - split_cost;
    if (!greater_zero(new_budget))
        new_budget = 0.0;
    if (budget_ > split_cost) {
        assert(is_leaf_);
        is_leaf_ = false;  /* Will now be a parent node */
        /* Check if min_block and max_block are minimum and maximum,
         * as it is possible that it is not updated yet (if this function
         * was called in extend_mondrian_block -> current node will
         * be updated afterwards */
        int feature_dim = mondrian_block_->get_feature_dim();
        arma::fvec min_block = mondrian_block_->get_min_block_dim();
        arma::fvec max_block = mondrian_block_->get_max_block_dim();
        arma::fvec max_block_sample = arma::max(max_block, sample.x);
        arma::fvec min_block_sample = arma::min(min_block, sample.x);
        arma::fvec dim_range = max_block_sample - min_block_sample;

        /* Sample split dimension */
        ///split_dim_ = sample_multinomial_scores(min_block);
        split_dim_ = sample_multinomial_scores(dim_range);
        /* Check if it is possible to introduce a split in current dimension */
        int count_sample_search = 0;
        int max_count_search = feature_dim;
        while (count_sample_search < max_count_search) {
            if (equal(min_block[split_dim_], max_block[split_dim_])) {
                split_dim_ = sample_multinomial_scores(min_block);
            } else {
                break;
            }
            count_sample_search += 1;
        }
        /* Compute left right statistics */
        std::pair<arma::fvec, arma::fvec> left_right_block;
        /* Special case if new samples lies not in current block
         * (not between min_block and max_block */
        split_loc_ = random.rand_uniform_distribution(min_block_sample[split_dim_], 
                max_block_sample[split_dim_]);
        if (max_block[split_dim_] < sample.x[split_dim_] ||
                equal(max_block[split_dim_],sample.x[split_dim_])) {
            min_block = mondrian_block_->get_max_block_dim();
            max_block = sample.x;
        } else if (min_block[split_dim_] > sample.x[split_dim_] ||
                equal(min_block[split_dim_],sample.x[split_dim_])) {
            max_block = mondrian_block_->get_min_block_dim();
            min_block = sample.x;
        } 
        if (settings_->debug) {
            cout << "min_block: " << min_block << endl;
            cout << "max_block: " << max_block << endl;
            cout << "split_dim: " << split_dim_ << endl;
        }
        /* Sample split location */
        /* Create new child nodes */
        /* Left side of the split */
        left_right_block = compute_left_right_statistics(split_dim_,
                split_loc_, sample.x,
                mondrian_block_->get_min_block_dim(),
                mondrian_block_->get_max_block_dim(), true);
        int tmp_depth = depth_+1;
        MondrianNode* left_child_node = new MondrianNode(num_classes_,
                feature_dim, new_budget, (*this),
                left_right_block.first, left_right_block.second,
                *settings_, tmp_depth);
        /* Right side of the split */
        left_right_block = compute_left_right_statistics(split_dim_,
                split_loc_, sample.x,
                mondrian_block_->get_min_block_dim(),
                mondrian_block_->get_max_block_dim(), false);
        MondrianNode* right_child_node = new MondrianNode(num_classes_,
                feature_dim, new_budget, (*this),
                left_right_block.first, left_right_block.second,
                *settings_, tmp_depth);
        id_left_child_node_ = left_child_node;
        id_right_child_node_ = right_child_node;

        if (sample.x[split_dim_] > split_loc_) {
            if (create_new_leaf) {
                MondrianNode* tmp_node = NULL;
                id_left_child_node_->init_update_posterior_node_incremental(*this);
                id_right_child_node_->init_update_posterior_node_incremental(
                        *tmp_node);
            } else {
                id_left_child_node_->init_update_posterior_node_incremental(*this);
                id_right_child_node_->init_update_posterior_node_incremental(*this);
            }
            /* Update new child node and check if node is "paused */
            id_right_child_node_->sample_mondrian_block(sample, true);
            id_right_child_node_->add_training_point_to_node(sample);
        } else {
            if (create_new_leaf) {
                MondrianNode* tmp_node = NULL;
                id_right_child_node_->init_update_posterior_node_incremental(*this);
                id_left_child_node_->init_update_posterior_node_incremental(
                        *tmp_node);
            } else {
                id_right_child_node_->init_update_posterior_node_incremental(*this);
                id_left_child_node_->init_update_posterior_node_incremental(*this);
            }
            /* Update new child node and check if node is "paused */
            id_left_child_node_->sample_mondrian_block(sample, true);
            id_left_child_node_->add_training_point_to_node(sample);
        }
                
    } else {
        is_leaf_ = true;
    }
}

/*
 * Extend mondrian block to include new training data
 */
void MondrianNode::extend_mondrian_block(const Sample& sample) {
    if (settings_->debug)
        cout << "### extend_mondrian_block: " << endl;

    float split_cost = 0.; /* On split_cost depends 
                             if a new split is introduced */
    /* 
     * Set new lower and upper boundary:
     *  - e_lower = max(l^x_j - x,0)
     *  - e_upper = min(x - u^x_j,0)
     */
    arma::fvec zero_vec(mondrian_block_->get_feature_dim(), arma::fill::zeros);
    arma::fvec tmp_min_block = mondrian_block_->get_min_block_dim();
    arma::fvec tmp_max_block = mondrian_block_->get_max_block_dim();

    arma::fvec e_lower = arma::max(
            zero_vec, (tmp_min_block - sample.x));
    arma::fvec e_upper = arma::max(
            zero_vec, (sample.x - tmp_max_block));
    /*
     * sample e (expo_param) from exponential distribution with rate
     * sum_d( e^l_d + e^u_d )
     */
    float expo_param = arma::sum(e_lower) + arma::sum(e_upper);

    /* Exponential distribution */
    if (!greater_zero(expo_param)) {
        split_cost = numeric_limits<float>::infinity();
    } else {
        /* Exponential distribution */
        split_cost = random.rand_exp_distribution(expo_param);
    }

    /* Check if all labels are identical */
    if (pause_mondrian()) {
        /* Try to extend a paused Mondrian (labels are not identical) */
        assert(is_leaf_);
        split_cost = numeric_limits<float>::infinity();
    }

    /*
     * Check if current point lies
     *  (1) Current point lies within Mondrian Block B^x_j
     *  (2) Current point lies outside block B^x_j
     */
    if (equal(split_cost, max_split_costs_) == true || 
            split_cost > max_split_costs_) {
        /* (1) Current point lies within Mondrian Block B^x_j */
        if (!is_leaf_) {    
            mondrian_block_->update_range_states(sample.x); 
            add_training_point_to_node(sample);
            /* 
             * Check split dimension/location to choose left or right node
             * and recurse on child 
             */
            if (sample.x[split_dim_] <= split_loc_) {
                assert(id_left_child_node_!=NULL);
                // Update density parameters
                bool left_split = true;
                update_density_parameters(left_split);
                // Recurse on child
                id_left_child_node_->extend_mondrian_block(sample);
            } else {
                assert(id_right_child_node_!=NULL);
                // Update density parameters
                bool left_split = false;
                update_density_parameters(left_split);
                // Recurse on child
                id_right_child_node_->extend_mondrian_block(sample);
            }
        } else {
            assert(is_leaf_);
            if (!check_if_same_labels(sample)) {
                sample_mondrian_block(sample);
            }
            /* Update after calling function sample_mondrian_block,
             * because of new node would take new boundary properties */
            mondrian_block_->update_range_states(sample.x);
            add_training_point_to_node(sample);
        }
    } else {
        /* (2) Current point lies outside block B^x_j */
        /* Initialize new parent node */
        int feature_dim = mondrian_block_->get_feature_dim();
        arma::fvec min_block = arma::min(mondrian_block_->get_min_block_dim(), 
                sample.x);
        arma::fvec max_block = arma::max(mondrian_block_->get_max_block_dim(),
                sample.x);
        MondrianNode* new_parent_node = new MondrianNode(num_classes_,
                feature_dim, budget_, *id_parent_node_,
                min_block, max_block, *settings_, depth_);
        /* Set "new_parent_node" as new parent of current node */
        /* Pass histogram of current node to new parent node */
        new_parent_node->init_update_posterior_node_incremental(*this, sample);
        /* 
         * Sample split dimension \delta, choosing d with probability 
         * proportional to e^l_d + d^u_d
         */
        arma::fvec feat_score = e_lower + e_upper;
        /* Problem can occur that min and max boundary value are the same
         * at a sampled split location -> solution: sample again until
         * it is different */
        int split_dim = sample_multinomial_scores(feat_score);
        /* Check if it is possible to introduce a split in current dimension */
        int max_sample_search = mondrian_block_->get_feature_dim();
        int count_sample_search = 0;
        while (count_sample_search < max_sample_search) {
            if (equal(min_block[split_dim_], max_block[split_dim_])) {
                split_dim_ = sample_multinomial_scores(min_block);
            } else {
                break;
            }
            count_sample_search += 1;
        }   
        float split_loc = 0.0;  /* Split location */
        /*
         * Sample split location \xi uniformly from interval 
         * [u^x_{j,\delta},x_\delta] if x_\delta > u^x_{j,\delta}
         * [x_delta, l^x_{j,\delta}] else
         */
        if (sample.x[split_dim] > mondrian_block_->get_max_block_dim()
                [split_dim]) {
            split_loc = random.rand_uniform_distribution(
                    mondrian_block_->get_min_block_dim()[split_dim],
                    sample.x[split_dim]);
        } else {
            split_loc = random.rand_uniform_distribution(sample.x[split_dim], 
                    mondrian_block_->get_min_block_dim()[split_dim]);
        }
        float new_budget = budget_ - split_cost;
        /* 
         * Insert a new node (j~) just above node j in the tree,
         * and a new leaf'', sibling to j 
         */
        bool is_left_node = false;
        arma::fvec new_child_block(mondrian_block_->get_feature_dim());
        if (sample.x[split_dim] > split_loc) {
            is_left_node = true;
            new_child_block = max_block;
        } else {
            new_child_block = min_block;
        }
        /* Grow Mondrian child node of the "outer Mondrian" */
        int new_depth = depth_ +1;
        MondrianNode* child_node = new MondrianNode(num_classes_,
                feature_dim, new_budget, *new_parent_node,
                new_child_block, new_child_block, *settings_, new_depth);
        /* Set child nodes of new created parent node ("new_parent_node") */
        new_parent_node->set_child_node(*child_node, (!is_left_node));
        new_parent_node->set_child_node(*this, is_left_node);
        new_parent_node->is_leaf_ = false;
        /* Set "new_parent_node" as new child node of current parent node */
        if (id_parent_node_ != NULL ) { /* root node */
            if (id_parent_node_->id_left_child_node_ == this) {
                id_parent_node_->set_child_node(*new_parent_node, true);
            } else {
                id_parent_node_->set_child_node(*new_parent_node, false);
            }
        }
        /* Set "new_parent_node" as new parent of current node */
        id_parent_node_ = new_parent_node;
        /*
         * Initialize posterior of new created child node 
         * (initialize histogram with zeros -> pointer = NULL) 
         */
        MondrianNode* tmp_node = NULL;
        child_node->init_update_posterior_node_incremental(*tmp_node, sample);

        child_node->sample_mondrian_block(sample);

        budget_ = new_budget;
        /* Update split cost of current and new parent node */ 
        new_parent_node->max_split_costs_ = split_cost;
        new_parent_node->split_loc_ = split_loc;
        new_parent_node->split_dim_ = split_dim;
        max_split_costs_ -= split_cost;
        update_depth();
    }
     
}

/**
 * Update parameters of density estimation
 */
void MondrianNode::update_density_parameters(bool left_split) {
    // Update the decision distribution parameters
    if (left_split){
        decision_distr_param_beta_++;
    }else{
        decision_distr_param_alpha_++;
    }
}


/*
 * Serialization of Mondrian block
 */
std::ostream & operator<<(std::ostream &os, const MondrianNode &mn) {
    return os << (*mn.num_classes_);
    /*
     * return os << mn.num_classes_ << mn.data_counter_ << mn.is_leaf_ <<
     *     mn.split_dim_ << mn.split_loc_ << mn.max_split_costs_ <<
     *     mn.budget_ << mn.pred_prob_ << mn.mondrian_block_ <<
     *     mn.id_left_child_node_ << mn.id_right_child_node_ <<
     *     mn.id_parent_node_ << mn.debug_;
     */
}

/*---------------------------------------------------------------------------*/
/*
 * Mondrian tree
 */
MondrianTree::MondrianTree(const mondrian_settings& settings, 
        const int& feature_dim) :
    num_classes_(0),
    settings_(&settings) {
    if (settings.debug)
        cout << "### Init Mondrian Tree " << endl;
    /* Root note has no parent node -> set NULL pointer */
    MondrianNode* null_parent_node = NULL;
    int depth = 0;
    float decision_distr_param_alpha = settings.decision_prior_hyperparam * 0.5;
    float decision_distr_param_beta = settings.decision_prior_hyperparam * 0.5;
    /* Initialize root node */
    root_node_ = new MondrianNode( &num_classes_, feature_dim, 
            std::numeric_limits<float>::infinity(),
            *null_parent_node, settings, depth,
            decision_distr_param_alpha, decision_distr_param_beta);
    
}

MondrianTree::~MondrianTree() {
    delete root_node_;  /* Delete root node */
}

/*
 * Print information of tree and recurse down to every node
 */
void MondrianTree::print_info() {
    cout << endl;
    cout << "----------------------------" << endl;
    cout << "Properties of current tree: " << endl;
    cout << "Number of classes: " << num_classes_ << endl;
    cout << "Data points:       " << data_counter_ << endl;
    cout << endl;
    root_node_->print_info();
}

/*
 * Update current data point
 */
void MondrianTree::update(Sample& sample) {
    /* Check if sample belongs to a new class */
    bool new_class = check_if_new_class(sample);
    if (new_class){
        update_class_numbers(sample);
    }
    ++data_counter_;  /* Update counter of data points */
    /* Start updating current sample at the root node of the tree */
    root_node_->update(sample);
    /* Check if there exists a new root node */
    root_node_ = root_node_->update_root_node();
    //root_node_->update_posteriors();
    //exit(EXIT_FAILURE);
     
}
/*
 * Predict class of current sample
 */
int MondrianTree::predict_class(Sample& sample, arma::fvec& pred_prob,
        mondrian_confidence& m_conf) {
    
    float prob_not_separated_yet = 1.;
    //arma::fvec pred_prob(num_classes_, arma::fill::zeros);
    int pred_class = root_node_->predict_class(sample, pred_prob,
            prob_not_separated_yet, m_conf); 
    if (settings_->debug) {
        cout << "pred class: " << pred_class << endl;
        cout << "prob: " << endl << pred_prob << endl;
    }
    return pred_class;
}

/*
 * Update number of classes
 *  - Increase variable num_classes_ +1
 *  - Update histograms all nodes
 */
void MondrianTree::update_class_numbers(Sample& sample) {
    if (settings_->debug)
        cout << "### update_class_numbers" << endl;
    /* +1 only works if first label = 0 */
    if (settings_->debug)
        cout << "num_classes: " << num_classes_ << endl;
    for (int i_new = num_classes_; i_new <= sample.y; i_new++) {
        ++num_classes_;  /* Increase number of classes */
    }
    /*
     * Update histogram of root node
     * -> will update all nodes of the tree
     */
    root_node_->add_new_class();
}

/*
 * Check if current data point belongs to a new/unknown class
 *  - at the moment it is only a simple request if the new data point
 *    belongs to a unknown class
 */
bool MondrianTree::check_if_new_class(Sample& sample) {
    bool new_class = false;
    /* 
     * If class label is greater than the number of classes
     * -> current sample is a new, unknown class
     * (+1 because class labels start with class 0)
     */
    if (sample.y + 1 > num_classes_) {
        new_class = true;
    }
    return new_class;
}


/*---------------------------------------------------------------------------*/
/*
 * Construct Mondrian forest
 */
MondrianForest::MondrianForest(const mondrian_settings& settings, 
                const int& feature_dim) :
    settings_(&settings) {
    MondrianTree* tree = NULL;
    for (int n_tree = 0; n_tree < settings.num_trees; n_tree++) {
        tree = new MondrianTree(settings, feature_dim);
        trees_.push_back(tree);
    }
}

MondrianForest::~MondrianForest() {
    /* Clear memory space */
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        delete trees_[n_tree];
    }
}

/*
 * Update current data point
 */ 
void MondrianForest::update(Sample& sample) {
    data_counter_++;

    /* Update all trees with current sample */
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        trees_[n_tree]->update(sample);
    }
}

/*
 * Predict class of current class
 */ 
int MondrianForest::predict_class(Sample& sample) {
    /* Go through all trees and calculate probability */
    //float expo_param = 1.0;
    mondrian_confidence m_conf;
    arma::fvec pred_prob = predict_probability(sample, m_conf);

    int pred_class = -1;  /* Predicted class of Mondrian forest */ 
    /* If all probabilies are the same -> return -2 */
    if (equal_elements(pred_prob)) {
        return -2;
    }

    float tmp_value = 0.;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > tmp_value) {
            tmp_value = pred_prob[i];
            pred_class = i;
        }
    }
    return pred_class;
}
/**
 * Predict class and return confidence
 */
pair<int, float> MondrianForest::predict_class_confident(Sample& sample) {
    pair<int, float> prediction (0, 0.0);
    
    /* Distance value that influence prediction */
    /* Init confidence values */
    mondrian_confidence m_conf;

    /* Go through all trees and calculate probability */
    arma::fvec pred_prob = predict_probability(sample, m_conf);

    int pred_class = -1;  /* Predicted class of Mondrian forest */
    float tmp_value = 0.;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > tmp_value) {
            tmp_value = pred_prob[i];
            pred_class = i;
        }
    }
    prediction.first = pred_class;

    /* Calculate confidence */
    float confidence = confidence_prediction(pred_prob, m_conf);
    prediction.second = confidence;
    return prediction;
}

void MondrianForest::print_info() {
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        trees_[n_tree]->print_info();
    }
}

/*
* Calculates probability of current sample
* (returns probability of all classes)
*/
arma::fvec MondrianForest::predict_probability(Sample& sample,
        mondrian_confidence& m_conf) {
    /* Go through all trees and calculate probability */
    arma::fvec pred_prob(trees_[0]->num_classes_, arma::fill::zeros);
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        arma::fvec tmp_pred_prob(trees_[0]->num_classes_, arma::fill::zeros);
        trees_[n_tree]->predict_class(sample, tmp_pred_prob, m_conf);
        pred_prob += tmp_pred_prob;
    }
    pred_prob = pred_prob / settings_->num_trees;

    return pred_prob;
}


/*
* Calculates confidence value
*/
float MondrianForest::confidence_prediction(arma::fvec& pred_prob,
        mondrian_confidence& m_conf) {
    float confidence = 0.0;
    
    /* Confidence: first best vs. second best */
    float first_class = 0.0;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > first_class) {
            first_class = pred_prob[i];
        }
    }
    float second_class = 0.0;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > second_class && pred_prob[i] < first_class) {
            second_class = pred_prob[i];
        }
    }
    if (greater_zero(first_class) && greater_zero(second_class)) {
        confidence = 1 - (second_class / first_class);
    } else {
        confidence = first_class;
    }
    
    /* Take mondrian confidence into account */
    int feature_dim = int(settings_->discount_param / settings_->discount_factor);
    float new_confidence = (1 - (5 * m_conf.distance/feature_dim)) * confidence;
    if (new_confidence < 0)
        new_confidence = 0;
    return new_confidence;
}
