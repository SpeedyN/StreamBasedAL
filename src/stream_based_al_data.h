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

#ifndef STREAM_BASED_AL_DATA_H_
#define STREAM_BASED_AL_DATA_H_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <set>
#include <armadillo>  /**< Matrix, vector library */
#include <algorithm>
#include "stream_based_al_random.h"

using namespace std;

/*---------------------------------------------------------------------------*/
/**
 * Class of a data point
 */
class Sample {
    public:
        arma::fvec x;  /**< Feature vector */
        int y;  /**< Class label */
};

/*---------------------------------------------------------------------------*/
/**
 * Result
 */
class Result {
  public:
    Result();
    ~Result(){};
    
    double training_time_;  /**< Save training time */
    double testing_time_;  /**< Save testing time */
    double accuracy_;  /**< Accuracy value of testing data */
    long int samples_used_for_training_;  /**< Number of samples that are
                                            used for training */
    vector<int> result_prediction_;  /**< Save all predictions of a
                                       Mondrian forest */
    vector<int> result_correct_prediction_;  /**< Save all correct classified
                                               samples */
    arma::Col<arma::uword> confidence_;  /**< Save all confidence values */
    /**< Save all false predicted confidence values */
    arma::Col<arma::uword> confidence_false_;
};

/*---------------------------------------------------------------------------*/
/**
 * Data set
 */
class DataSet {
    public:
        
        DataSet();
        DataSet(const bool random, const bool sort_classes,
                const bool load_iterative_);
        ~DataSet();
        /**
         * Load data points and labels from file
         *
         * @param filename: file with data points and labels !! TODO:
         * @param x_filename: file with data points only
         * @param y_filename: file with labels only
         */
        void load(const string& filename);
        void load(const string& x_filename, const string& y_filename,
                bool add_points = false);
        /**
         * Returns next sample
         */
        Sample get_next_sample();
        /**
         * Set position of "samples_" back to zero
         */
        inline void reset_position() {sample_pos_ = 0;};

        /* Class properties */
        long int num_samples_;  /**< Number of samples in current file */
        int feature_dim_;  /**< Feature dimension */
        int num_classes_;  /**< Number of classes */

    private:
        /* Properties load data */
        const bool random_;  /**< Will select samples randomly */
        const bool sort_classes_;  /**< Samples are sorted after classes */
        const bool load_iterative_;  /**< Load data points one after another */
        int sample_pos_;  /**< Saves currnet position of vector "samples_" */
        vector<Sample> samples_;  /**< Vector with all samples */
        vector<long int> x_file_position_;  /**< Vector with absolute position
                                            of the beginning of a lint */
        vector<long int> y_file_position_;  /**< Vector with absolute position
                                            of the beginning of a lint */
        string x_filename_;  /**< Filename of data points */
        string y_filename_;  /**< Filename of labels */
        ifstream x_file_;  /**< Open stream of data points */
        ifstream y_file_;  /**< Open stream of data labels */
        vector<long int> rand_vec_;  /**< To get access to x_file_position
                                       and y_file_position randomly */
        bool add_points_;  /**< Option to include data points to "samples_"
                             afterwards */
        /**
         * Load complete dataset into memory
         */
        void load_complete_dataset(const string& x_filename,
                const string& y_filename);
        /**
         * Load / Open file to jump to defined position of the file
         * afterwards
         */
        void load_dataset_iteratively(const string& x_filename,
               const string& y_filename); 
        /**
         * Create file with the position of each line 
         */
        void create_position_file(const string& file);
        /**
         * Open file with the position of each line 
         */
        void open_position_file(const string& file);
        /**
         * Get sample from defined position of file
         */
        void get_sample_from_file(Sample& sample);

        
};
/*
 * Function inserts current point with decreasing order
 * (depends on buffer.second)
 */
inline void insert_sort(list<pair<Sample, float> >& buffer,
        pair<Sample, float>& cur_sample) {
    if (buffer.size() < 1) {
        buffer.push_back(cur_sample);
    } else {
        list<pair<Sample, float> >::iterator it = buffer.begin();
        bool insert_sample = false;
        for(; it != buffer.end(); it++) {
            if ((*it).second > cur_sample.second) {
                buffer.insert(it, cur_sample);
                insert_sample = true;
            }
            if (insert_sample)
                break;
        }
        if (!insert_sample)
            buffer.push_back(cur_sample);
    }
}
#endif /* STREAM_BASED_AL_DATA_H_ */
