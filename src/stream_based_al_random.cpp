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

#include "stream_based_al_random.h"

/*---------------------------------------------------------------------------*/
bool RandomGenerator::seed_flag_ = false;
base_generator_type RandomGenerator::generator( init_seed() );

/*---------------------------------------------------------------------------*/
RandomGenerator::RandomGenerator() :
    uni_dist(0.0,1.0),
    uni_gen(generator ,uni_dist) {
    if (!seed_flag_) {
        generator.seed(static_cast<unsigned int>(init_seed()));
        seed_flag_ = true;
    }
}

float RandomGenerator::rand_uniform_distribution() {
    return uni_gen();
}

float RandomGenerator::rand_uniform_distribution(
        float min_value, float max_value) {
    if (equal(min_value,max_value)) {
        max_value += eps;
        //cout << "[WARNING]: - rand_uniform_distribution: min_value == max_value" << endl;
    }
    float rand_value = min_value + (max_value - min_value) * 
        rand_uniform_distribution();

    return rand_value;
}

float RandomGenerator::rand_uniform_distribution(
        float min_value, float max_value, bool& equal_values) {
    if (equal(min_value,max_value)) {
        max_value += eps;
        //cout << "[WARNING]: - rand_uniform_distribution: min_value == max_value" << endl;
        equal_values = true;
    } else {
        equal_values = false;
    }
    float rand_value = min_value + (max_value - min_value) * 
        rand_uniform_distribution();

    return rand_value;
}

float RandomGenerator::rand_exp_distribution(float lambda) {
    if (equal(lambda,0.0) || !greater_zero(lambda) ) {
        lambda = 1;
    }
    boost::exponential_distribution<float> exp_dist(lambda);
    boost::variate_generator<base_generator_type&,
        boost::exponential_distribution<float> > exp_gen(generator, exp_dist);
    return exp_gen();
}

float RandomGenerator::rand_beta_distribution(float alpha, float beta) {
    if (equal(alpha,0.0) || !greater_zero(alpha) ) {
        alpha = 1;
    }
    if (equal(beta,0.0) || !greater_zero(beta) ) {
        beta = 1;
    }
    boost::random::beta_distribution<> beta_dist(alpha, beta);
    boost::variate_generator<base_generator_type&,
        boost::random::beta_distribution<> > beta_gen(generator, beta_dist);
    return beta_gen();
}
