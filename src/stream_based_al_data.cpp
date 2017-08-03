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

#include "stream_based_al_data.h"

/*---------------------------------------------------------------------------*/
/*
 * Result
 */
Result::Result() :
  training_time_(0.0),
  testing_time_(0.0),
  accuracy_(0.0),
  samples_used_for_training_(0),
  confidence_(20, arma::fill::zeros),
  confidence_false_(20, arma::fill::zeros) {
}

/*---------------------------------------------------------------------------*/
/*
 * Data set
 */
DataSet::DataSet() :
    random_(false),
    sort_classes_(false),
    load_iterative_(false),
    sample_pos_(0),
    add_points_(false) {
}

DataSet::DataSet(const bool random, const bool sort_classes, 
        const bool load_iterative) :
    random_(random),
    sort_classes_(sort_classes),
    load_iterative_(load_iterative),
    sample_pos_(0),
    add_points_(false) {

}

DataSet::~DataSet() {
    if (load_iterative_) {
        x_file_.close();
        y_file_.close();
    }
}

/* TODO:
 * Load data points and labels from file (libsvm)
 */
void DataSet::load(const string& filename) {
    ifstream file(filename.c_str(), ios::binary);
    if (!file) {
        cout << "Could not open input file " << filename << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Loading data file: " << filename << " ... " << endl;
    samples_.clear();

    //TODO:

    file.close();
}

/*
 * Load data points and labels that are in different files
 */
void DataSet::load(const string& x_filename, const string& y_filename,
        bool add_points) {
    x_filename_ = x_filename;
    y_filename_ = y_filename;
    add_points_ = add_points;
    reset_position();
    if (load_iterative_) {
        if (add_points) {
            cout << "[ERROR] - DataSet.load: option 'add' and 'load_iterative' is not possible" << endl;
            exit(EXIT_FAILURE);
        }
        x_file_.close();
        y_file_.close();
        load_dataset_iteratively(x_filename, y_filename);
        open_position_file(x_filename); 
    } else {
        if (add_points)
            sample_pos_ = 0;
        load_complete_dataset(x_filename, y_filename);
    }
}

/*
 * Returns next sample
 */
Sample DataSet::get_next_sample() {
    Sample cur_sample;
    if (sample_pos_ < num_samples_) {
        if (load_iterative_) {
            get_sample_from_file(cur_sample);
            sample_pos_++; 
        } else {
            cur_sample = samples_[sample_pos_];
            sample_pos_++;
        }
    } else {
        cout << "[ERROR] - DataSet.get_next_sample(): end of file" << endl;
        exit(EXIT_FAILURE);
    }
    return cur_sample;
}

/*
 * Load complete dataset into memory
 */
void DataSet::load_complete_dataset(const string& x_filename,
        const string& y_filename) {
    /* Try to open files */
    ifstream xfp(x_filename.c_str(), ios::binary);
    if (!xfp) {
        cout << "Could not open input file " << x_filename << endl;
        exit(EXIT_FAILURE);
    }
    ifstream yfp(y_filename.c_str(), ios::binary);
    if (!yfp) {
        cout << "Could not open input file " << y_filename << endl;
        exit(EXIT_FAILURE);
    }
    cout << endl;
    cout << "Loading data file: " << x_filename << " ... " << endl;
    cout << "Loading data file: " << y_filename << " ... " << endl;

    /* Reading the header (first line of file)*/
    int tmp;
    long int tmp_samples;
    xfp >> tmp_samples;
    num_samples_ = tmp_samples;
    xfp >> feature_dim_;
    yfp >> tmp;
    if (tmp != tmp_samples) {
        cout << "Number of samples in data and labels file is different" << endl;
        exit(EXIT_FAILURE);
    }
    yfp >> tmp;
    /* Delete list with data points */
    if (!add_points_)
        samples_.clear();
    set<int> labels;
    /* Going through complete files */
    for (int n_samp = 0; n_samp < num_samples_; n_samp++) {
        Sample sample;
        sample.x = arma::fvec(feature_dim_);
        yfp >> sample.y;
        labels.insert(sample.y);
        for (int n_feat = 0; n_feat < feature_dim_; n_feat++) {
            xfp >> sample.x(n_feat);
        }
        samples_.push_back(sample);
    }
    xfp.close();
    yfp.close();
    num_classes_ = labels.size();

    if (random_) {
        srand(init_seed());
        random_shuffle(samples_.begin(), samples_.end());
    }
}


/*
 * Load complete dataset into memory
 */
void DataSet::load_dataset_iteratively(const string& x_filename,
        const string& y_filename) {
    /* Try to open files and read header */
    x_file_.open(x_filename.c_str(), ios::binary);
    if (!x_file_) {
        cout << "Could not open input file " << x_filename << endl;
        exit(EXIT_FAILURE);
    }
    y_file_.open(y_filename.c_str(), ios::binary);
    if (!y_file_) {
        cout << "Could not open input file " << y_filename << endl;
        exit(EXIT_FAILURE);
    }
    cout << endl;
    cout << "Loading data file: " << x_filename << " ... " << endl;
    cout << "Loading data file: " << y_filename << " ... " << endl;

    /* Reading the header (first line of file)*/
    int tmp;
    x_file_ >> num_samples_;
    x_file_ >> feature_dim_;
    y_file_ >> tmp;
    if (tmp != num_samples_) {
        cout << "Number of samples in data and labels file is different" << endl;
        exit(EXIT_FAILURE);
    }
    y_file_ >> tmp;
    /* Initialize random vector */
    if (random_) {
        for (long int i = 0; i < num_samples_; i++) {
            rand_vec_.push_back(i);
        }
        srand(init_seed());
        random_shuffle(rand_vec_.begin(), rand_vec_.end());
    }
}

/*
 * Create file with the position of each line 
 */
void DataSet::create_position_file(const string& file) {
    cout << endl;
    cout << "Trying to create file with all line positions ..." << endl;

    int pos = file.find(".");
    string file_tmp = file.substr(0, pos);
    string x_filename = file_tmp + ".pos_data";
    string y_filename = file_tmp + ".pos_labels";

    ofstream x_num_file(x_filename.c_str(), ios::binary);
    ofstream y_num_file(y_filename.c_str(), ios::binary);
    
    /* Try to open files */
    ifstream xfp(x_filename_.c_str(), ios::binary);
    if (!xfp) {
        cout << "Could not open input file " << x_filename_ << endl;
        exit(EXIT_FAILURE);
    }
    ifstream yfp(y_filename_.c_str(), ios::binary);
    if (!yfp) {
        cout << "Could not open input file " << y_filename_ << endl;
        exit(EXIT_FAILURE);
    }

    /* Reading the header (first line of file)*/
    int tmp;
    xfp >> num_samples_;
    xfp >> feature_dim_;
    yfp >> tmp;
    if (tmp != num_samples_) {
        cout << "Number of samples in data and labels file is different" << endl;
        exit(EXIT_FAILURE);
    }
    yfp >> tmp;

    x_num_file << xfp.tellg();
    x_num_file << "\n";
    y_num_file << yfp.tellg();
    y_num_file << "\n";
    /* Going through complete files */
    for (int n_samp = 0; n_samp < num_samples_; n_samp++) {
        Sample sample;
        sample.x = arma::fvec(feature_dim_);
        yfp >> sample.y;
        y_num_file << yfp.tellg();
        y_num_file << "\n";
        for (int n_feat = 0; n_feat < feature_dim_; n_feat++) {
            xfp >> sample.x(n_feat);
        }
        x_num_file << xfp.tellg();
        x_num_file << "\n";
    }
    xfp.close();
    yfp.close();
    
    x_num_file.close();
    y_num_file.close();

}

/*
 * Open file with position of each line
 */
void DataSet::open_position_file(const string& file) {
     
    int pos = file.find(".");
    string file_tmp = file.substr(0, pos);
    string x_filename = file_tmp + ".pos_data";
    string y_filename = file_tmp + ".pos_labels";
    cout << endl;
    cout << "Loading data file: " << x_filename << " ... " << endl;
    cout << "Loading data file: " << y_filename << " ... " << endl;

    ifstream x_num_file(x_filename.c_str(), ios::binary);
    if (!x_num_file) {
        cout << "Could not open input file " << x_filename << endl;
        create_position_file(file);
        x_num_file.open(x_filename.c_str(), ios::binary);
        if (!x_num_file)
            exit(EXIT_FAILURE);
    }
    ifstream y_num_file(y_filename.c_str(), ios::binary);
    if (!x_num_file) {
        cout << "Could not open input file " << y_filename << endl;
        exit(EXIT_FAILURE);
    }
    int cur_num;
    for (int n_samp = 0; n_samp < num_samples_; n_samp++) {
        x_num_file >> cur_num;
        x_file_position_.push_back(cur_num);
        y_num_file >> cur_num;
        y_file_position_.push_back(cur_num);
    }
    
    x_num_file.close();
    y_num_file.close();
}

/*
 * Get sample from defined position of file
 */
void DataSet::get_sample_from_file(Sample& sample) {
    long int position = 0;
    if (random_) {
        position = rand_vec_[sample_pos_];
    } else {
        position = sample_pos_;
    }
    y_file_.seekg(y_file_position_[position]);
    x_file_.seekg(x_file_position_[position]); 
    /* Get current line of file */
    sample.x = arma::fvec(feature_dim_);
    y_file_ >> sample.y;
    for (int n_feat = 0; n_feat < feature_dim_; n_feat++) {
        x_file_ >> sample.x(n_feat);
    }
    
}
