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

#ifndef STREAM_BASED_AL_FOREST_H_
#define STREAM_BASED_AL_FOREST_H_

/* Evaluate assertion */
//#define NDEBUG /* Comment out if no assertion is needed */
#include <assert.h> 

#include <list>
#include <algorithm>  /* Used for count elements in vector */
#include <armadillo>  /* Matrix, vector library */
#include "stream_based_al_utilities.h"
#include "stream_based_al_data.h"
#include <limits>

/* Boost libraries for serialization */
#include <boost/archive/tmpdir.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/assume_abstract.hpp>

/*---------------------------------------------------------------------------*/
// TODO: serialization only works for Mondrian Block
/**
 * Function to serialize arma::fvec
 */
namespace boost {
    namespace serialization {
        template<class Archive>
            void serialize(Archive & ar, arma::fvec& v,
                    const unsigned int version) {
                const size_t data_size = v.size() * sizeof(float);
                ar & boost::serialization::make_array(&v[0], data_size);
            }
    }
}

/*---------------------------------------------------------------------------*/
/**
 * Settings to initialize a Mondrian tree
 *
 * @param num_trees         : number of trees in a Mondrian forest
 * @param init_budget       : init budget for lifetime parameter
 * @param discount_factor   :
 * @param debug             : set debug mode
 */
struct mondrian_settings {
    int num_trees;
    float init_budget;
    float discount_factor;
    float discount_param;
    float decision_prior_hyperparam;
    bool debug;
    int max_samples_in_one_node;
};
/*---------------------------------------------------------------------------*/
/**
 * Confidence values of a Mondrian tree
 */
struct mondrian_confidence {
    int number_of_points;
    float density;
    float distance;
};
/*---------------------------------------------------------------------------*/
/**
 * Defines a Mondrian block
 *
 * @param feature_dim_    : Dimension of feature vector
 * @param min_block_dim_  : Dimension-wise min of training data in current block
 * @param max_block_dim_  : Dimension-wise max of training data in current block
 * @param sum_range_dim_  : Sum of range of all dimensions
 */
class MondrianBlock {

    public:
        /**
         * Construct mondrian block
         */
        MondrianBlock() : feature_dim_(1) {};
        MondrianBlock(const int& feature_dim,
                const mondrian_settings& settings);
        MondrianBlock(const int& feature_dim,  /* Feature dimension */
                arma::fvec& min_block_dim,  /* Lower block boundary */
                arma::fvec& max_block_dim,  /* Upper block boundary */
                const mondrian_settings& settings);

        ~MondrianBlock();
        /**
         * Get feature dimension
         */
        inline int get_feature_dim();
        /**
         * Get dimension range
         */
        float get_dim_range(const arma::fvec& cur_sample);
        /**
         * Get lower block boundary
         */
        inline arma::fvec get_min_block_dim();
        /**
         * Get upper block boundary
         */
        inline arma::fvec get_max_block_dim();
        /**
         * Update minimum and maximum of training data at this block
         */
        void update_range_states(const arma::fvec& cur_min_dim, 
                const arma::fvec& cur_max_dim);
        /**
         * Update minimum and maximum of training data at this block
         */
        void update_range_states(const arma::fvec& cur_point);
        /**
         * Get sum_dim_range
         */
        inline float get_sum_dim_range();

    private:
        /* Set functions ostream and serialization as friend */
        friend std::ostream & operator<<(std::ostream &os,
                const MondrianBlock &mb);
        friend class boost::serialization::access;  /**< Serialization */

        const int feature_dim_;  /**< Feature dimension. */
        float sum_dim_range_;  /**< Sum of range of all dimensions */
        /**
         * Every Mondrian block has a lower and upper boundary
         * in each dimension.
         */
        arma::fvec min_block_dim_;  /**< Dimension-wise minimum of training 
                                       data in current block (left border) */
        arma::fvec max_block_dim_;  /**< Dimension-wise maximum of training 
                                       data in current block (right border) */
        bool debug_;  /**< Debug mode */
        /**
         * Calculate sum of all dimension = sum(max_block_dim_ - min_block_dim_)
         */
        void update_sum_dim_range();
        /**
         * Serialization of Mondrian block
         */
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            //ar & boost::serialization::make_array(&min_block_dim_, feature_dim_);
            ar & const_cast<int &> (feature_dim_);
            ar & sum_dim_range_;
            min_block_dim_.resize(feature_dim_);
            max_block_dim_.resize(feature_dim_);
            ar & min_block_dim_;
            ar & max_block_dim_;
            ar & debug_;
        }

};

/*
 * Return feature dimension
 */
inline int MondrianBlock::get_feature_dim() {
    return feature_dim_;
}

/*
 * Get lower block boundary
 */
inline arma::fvec MondrianBlock::get_min_block_dim() {
    return min_block_dim_;
}
/*
 * Get upper block boundary
 */
inline arma::fvec MondrianBlock::get_max_block_dim() {
    return max_block_dim_;
}

/*
 * Get sum_dim_range
 */
inline float MondrianBlock::get_sum_dim_range() {
    return sum_dim_range_;
}

/*---------------------------------------------------------------------------*/
/**
 * Defines a Mondrian node of a mondrian tree with one mondrian block
 *
 * @param id_left_       : Id of left child
 * @param id_right_      : Id of right child
 * @param id_parent_     : Id of parent node
 * @param is_leaf_       : Boolen variable to indicate if current node 
 *                          is a leafnode
 * @param budget_        : Remaining lifetime for subtree - time of split 
 *                          of parent
 *                          NOTE: time of split of parent of root node is 0
 * @param max_split_cost_: Maximum split cost for a node ist time of 
 *                          split of node - time of split of parent and
 *                          is drawn from an exponential
 * @param count_labels_  : Stores histogram of labels at each node
 */
class MondrianNode {
    public:
        /**
         * Construct tree node
         */
        MondrianNode() : settings_(NULL) {};
        MondrianNode(int* num_classes, const int& feature_dim,
                const float& budget, MondrianNode& parent_node,
                const mondrian_settings& settings, int& depth,
                float& expected_prob_mass,
                float& decision_distr_param_alpha,
                float& decision_distr_param_beta);
        /**
         * Construct tree node with given values of boundaries of
         * the Mondrian block
         * @param parent_node   : pointer to parent node
         * @param min_block_dim : Lower boundary of Mondrian block
         * @param max_block_dim : Upper boundary of Mondrian block
         */
        MondrianNode(int* num_classes, const int& feature_dim, 
                const float& budget, MondrianNode& parent_node,
                arma::fvec& min_block_dim, arma::fvec& max_block_dim,
                const mondrian_settings& settings, int& depth,
                float& expected_prob_mass,
                float& decision_distr_param_alpha,
                float& decision_distr_param_beta);
        /**
         * Construct tree node with given values of boundaries of
         * the Mondrian block and one existing child node
         *
         * @param parent_node       : Pointer to parent node
         * @param left_child_node   : Pointer to left child node
         * @param right_child_node  : Pointer to right child node
         * @param min_block_dim     : Lower boundary of Mondrian block
         * @param max_block_dim     : Upper boundary of Mondrian block
         */
        MondrianNode(int* num_classes, const int& feature_dim,
                const float& budget, MondrianNode& parent_node, 
                MondrianNode& left_child_node, MondrianNode& right_child_node,
                arma::fvec& min_block_dim, arma::fvec& max_block_dim,
                const mondrian_settings& settings, int& depth,
                float& expected_prob_mass,
                float& decision_distr_param_alpha,
                float& decision_distr_param_beta);
        ~MondrianNode();
        /**
         * Print information of current node
         */
        void print_info();
        /**
         * Update histogram with additional class and 
         * increase class histogram +1 (add new class)
         * - Updates child nodes too
         */
        void add_new_class();
        /**
         * Predict class of current sample
         */
        int predict_class(Sample& sample, arma::fvec& pred_prob,
                float& prob_not_separated_yet, mondrian_confidence& m_conf);
        /**
         * Return address of root node
         *
         * If a new node is set above the current node, the root node can
         * change. The problem is that if the tree is updated, the class
         * "MondrianTree" will start at the root node but points to another
         * node. Therefore, this function finds the root of the tree
         */
        MondrianNode* update_root_node();
        /**
         * Update current data sample
         *
         * @param sample    : Current data point
         * @param new_class : Defines if class of current data point is known
         *                    If true:
         *                       1. update num_classes_ (number of classes)
         *                       2. update count_labels_ (label histogram)
         */
        void update(const Sample& sample);

    private:
        /**< Set functions ostream and serialization as friend */
        friend std::ostream & operator<<(std::ostream &os,
                const MondrianNode &mn);
        friend class boost::serialization::access;  /**< Serialization */

        int* num_classes_;  /**< Number of classes */
        float data_counter_;  /**< Count data points */
        bool is_leaf_;  /**< Boolean variable to indicate if current node
                          is a leaf node */
        int split_dim_; /**< Split dimension (feat_id_chosen) */
        float split_loc_; /**< Split location (split_chosen) */
        float max_split_costs_; /**< Maximum split cost for a node ist time of
                                   split of node - time of split of parent and
                                   is drawn from an exponential */
        arma::Col<arma::uword> count_labels_;  /**< Stores histogram of lavels
                                                 at each node */
        float budget_;  /**< Represent remaining budget of current node */
        arma::fvec pred_prob_;
        MondrianBlock* mondrian_block_;  /**< Pointer to mondrian block */
        /**
         * Pointer to child (left, right) and parent node
         */
        MondrianNode* id_left_child_node_; /**< Pointer to left child node */
        MondrianNode* id_right_child_node_; /**< Pointer to right child node */
        MondrianNode* id_parent_node_; /**< Pointer to parent node */
        const mondrian_settings* settings_;  /**< Mondrian settings */
        int depth_;  /**< Current depth of node in the tree */
        float expected_prob_mass_; /**< Expected probability mass assigned
                                   to the node given by the decision random
                                   variables on the path to this node. */
        float decision_distr_param_alpha_; /**< Parameters of the estimated */
        float decision_distr_param_beta_;  /*   decision distribution at this node.*/
        bool debug_;  /**< Set debug mode */

        static RandomGenerator random;  /**< Random generator */
        /**
         * Checks if all labels in a node are identical
         * - go through vector count_labels_ and check 
         *   if only one element is > 1
         */
        bool check_if_same_labels();
        /**
         * Checks if all labels and the current point of a node are identical
         * @param sample    : Current data point
         */
        bool check_if_same_labels(const Sample& sample);
        /**
         * Checks if a Mondrian block should be paused
         * - pause if all labels in a node are identical
         */
        bool pause_mondrian();
       /**
        * Update statistic (posterior) of current node
        * @param sample     : Current data point
        */
       void update_posterior_node_incremental(const Sample& sample);
       /**
        * Initialize update posterior node
        * 
        * @param node_id    : Id of Mondrian node to copy histogram from
        * @param sample     : Current data point
        */
       void init_update_posterior_node_incremental(MondrianNode& node_id, 
               const Sample& sample);
       void init_update_posterior_node_incremental(MondrianNode& node_id);
       /**
        * Add a training data point to current node
        */
       void add_training_point_to_node(const Sample& sample);
        /**
         * Compute left right statistic
         */
        std::pair<arma::fvec, arma::fvec> compute_left_right_statistics( 
                int& split_dim, float& split_loc, const arma::fvec& sample_x,
                arma::fvec min_cur_block, arma::fvec max_cur_block,
                bool left_split);
        /**
         * Pass child node and set variable "is_leaf_" to false
         *
         * @param child_node    : Pointer to child node
         * @param is_left_node  : Defines right (false) or left (true) child node
         */
        void set_child_node(MondrianNode& child_node, bool is_left_node);
        /**
         * Get counts to calculate posterior inference (Chinese restaurant)
         * @param num_tables_k  : number of class k
         * @param num_customers : training data points
         * @param num_tables    :
         */
        void get_counts(int& num_tables_k, int& num_customers, int& num_tables);
        /**
         * Compute prior mean
         */
        arma::fvec get_prior_mean(arma::fvec& pred_prob_par);
        arma::fvec get_prior_mean();
        /**
         * Compute posterior mean
         */
        arma::fvec compute_posterior_mean_normalized_stable(
                arma::Col<arma::uword>& cnt, float& discount,
                arma::fvec& base);
        void update_depth();
        /**
         * Update split_cost
         */
        void sample_mondrian_block(const Sample& sample,
                bool create_new_leaf = false);
        /**
         * Extend mondrian block to include new training data
         */
        void extend_mondrian_block(const Sample& sample);
};

/*---------------------------------------------------------------------------*/
/**
 * Defines a Mondrian tree
 * 
 * @param settings_     : Mondrian settings
 * @param root_node_    : Pointer to root node of the tree
 * @param num_classes_  : Number of classes (different labels)
 * @param data_counter_ : Counts all data points that pass by
 */
class MondrianTree {
    public:
        /**
         * Construct mondrian tree
         *
         * @param settings      : Mondrian settings
         * @param feature_dim   : Dimension of feature vector
         */
        MondrianTree(const mondrian_settings& settings,
                const int& feature_dim);

        ~MondrianTree();

        int num_classes_;  /**< Number of classes (different labels) */
        /**
         * Print information of tree and every node
         */
        void print_info();

        /**
         * Update current data point
         */ 
        void update(Sample& sample);
        /**
         * Predict class of current sample
         */
        int predict_class(Sample& sample, arma::fvec& pred_prob,
                mondrian_confidence& m_conf);

    private:

        float data_counter_;  /**< Count data points */
        MondrianNode* root_node_;  /**< Pointer to root node */
        const mondrian_settings* settings_;  /**< Settings of Mondrian forest */
        /**
         * Update number of classes 
         *  - increase variable num_classes_ +1
         */
        void update_class_numbers(Sample& sample);
        /*
         * Check if current data point belongs to a new/unknown class
         *  - at the moment it is only a simple request if the new data point
         *    belongs to a unknown class
         */
        bool check_if_new_class(Sample& sample);
};

/*---------------------------------------------------------------------------*/
/**
 * Defines a Mondrian forest -> defined number of Mondrian trees
 *
 * @param data_counter_ : Counts all data points that pass by
 * @param trees_        : Vector contains all Mondrian trees
 * @param settings_     : Settings of Mondrian forest
 */
class MondrianForest {
    public:
        /**
         * Construct mondrian tree
         */
        MondrianForest(const mondrian_settings& settings, 
                const int& features_dim);

        ~MondrianForest();
        /**
         * Update current data point
         */ 
        void update(Sample& sample);
        /**
         * Predict class of current class
         */ 
        int predict_class(Sample& sample);
        /**
         * Predict class and return confidence
         */
        pair<int, float> predict_class_confident(Sample& sample);

        void print_info();
        
    private:
        float data_counter_;  /**< Count incoming data points */
        vector<MondrianTree*> trees_;  /**< Save all Mondrian trees */
        const mondrian_settings* settings_;  /**< Setting of Mondrian forest */
        /*
         * Calculates probability of current sample
         * (returns probability of all classes)
         */
        arma::fvec predict_probability(Sample& sample,
                mondrian_confidence& m_conf);
        /*
         * Calculates confidence value
         */
        float confidence_prediction(arma::fvec& prediction,
                mondrian_confidence& m_conf);
        
};

#endif /* STREAM_BASED_AL_FOREST_H_ */
