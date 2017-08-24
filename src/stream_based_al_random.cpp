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
/* Instantiate global random number generator */
RandomGenerator rng;

/*---------------------------------------------------------------------------*/
RandomGenerator::RandomGenerator() :
    generator(init_seed()),
    uni_dist(0.0,1.0),
    uni_gen(generator ,uni_dist) {
}

unsigned int RandomGenerator::init_seed() {
    ifstream devFile("/dev/urandom", ios::binary);
    unsigned int outInt = 0;
    char tempChar[sizeof(outInt)];
    
    devFile.read(tempChar, sizeof(outInt));
    outInt = atoi(tempChar);
    
    devFile.close();
    
    struct timeval TV;
    gettimeofday(&TV, NULL);
    unsigned int seed = (unsigned int) TV.tv_sec * TV.tv_usec + getpid() + outInt;
    return seed;
}

void RandomGenerator::set_seed(unsigned int new_seed){
    generator.seed(new_seed);
    uni_gen.engine().seed(new_seed);
    uni_gen.distribution().reset();
}

float RandomGenerator::rand_uniform_distribution() {
    return uni_gen();
}

float RandomGenerator::rand_uniform_distribution(
        float min_value, float max_value) {
    if (equal(min_value,max_value)) {
        max_value += eps; //TODO: find way around this, causes bugs down the road
        //cout << "[WARNING]: - rand_uniform_distribution: min_value == max_value" << endl;
    }
    float rand_value = min_value + (max_value - min_value) * 
        rand_uniform_distribution();

    return rand_value;
}

float RandomGenerator::rand_uniform_distribution(
        float min_value, float max_value, bool& equal_values) {
    if (equal(min_value,max_value)) {
        max_value += eps; //TODO: find way around this, causes bugs down the road
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
