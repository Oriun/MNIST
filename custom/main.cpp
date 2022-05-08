#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <array>
#include <climits>
#include <map>
#include <thread>
#include <future>

#include "dataset.cpp"
// #include "result.cpp"

using namespace std;

using pixel = unsigned char;
using matrix = array<pixel, 784>;
using image_set = vector<matrix>;
using label_set = vector<char>;
using dist = unsigned long long int;

struct prediction_item
{
    prediction_item() : label(-1), distance(ULLONG_MAX){};
    prediction_item(short a, dist b) : label(a), distance(b){};
    short label;
    dist distance;
};

dist euclidian_distance(const matrix &A, const matrix &B)
{
    if (A.size() != B.size())
    {
        throw runtime_error("Matrices doesn't have the same type");
    }
    dist k(0);
    for (int i = 0; i < A.size(); i++)
    {
        k += pow((A[i] - B[i]), 2);
    }
    return k;
}

short choose_from_predictions(const vector<prediction_item> &list)
{
    dist max = list.back().distance;

    map<short, vector<dist>> s;
    for (prediction_item k : list)
    {
        s[k.label].push_back(max - k.distance);
    }

    map<short, dist> t;
    for (const auto &k : s)
    {
        dist g = 0;
        for (dist h : k.second)
        {
            g += h;
        }
        g /= k.second.size();
        t[k.first] += g;
    }

    short res = 0;

    for (const auto &k : t)
    {
        if (k.second > t[res])
        {
            res = k.first;
        }
    }
    cout << res << endl;
    return res;
}

short predict(dataset train_set, const matrix &matrix_to_test, unsigned short int neighbors)
{
    cout << "Predicting... ";
    vector<prediction_item> list(neighbors);

    for (int l = 0; l < train_set.imgs.size(); l++)
    {
        const matrix &D = train_set.imgs[l];
        dist d = euclidian_distance(D, matrix_to_test);
        for (unsigned short int i = 0; i < neighbors; i++)
        {
            if (list[i].distance > d)
            {
                prediction_item p(train_set.labels[l], d);
                list.insert(begin(list) + i, p);
                list.pop_back();
                break;
            }
        }
    }

    return choose_from_predictions(list);
}

int main(int argc, char **argv)
{
    int neighbors = 5;
    if (argc > 1)
    {
        neighbors = stoi(argv[1]);
    }
    cout << "Starting" << endl;

    cout << "Loading Train Dataset __________________________" << endl;
    dataset train_set = load(false);
    cout << "Loading Test _Dataset _________________________" << endl;
    dataset test_set = load(true);

    // for (int g = 0; g < 100; g++)
    // {
    //     for(int g___ = 0; g___ < 784; g___++){
    //         cout << setfill('0') << setw(2) << hex << +test_set.imgs[g][g___];
    //         if((g___ %28) == 0) cout << endl;
    //     }
    //     cout << "label : " << +test_set.labels[g] << endl;
    //     char g_;
    //     cin >> g_;
    // }

    int errors(0);
    int success(0);
    for (int i = 0; i < test_set.imgs.size(); i++)
    {
        matrix k = test_set.imgs[i];
        cout << "Test n°" << (errors + success) << " expecting " << +test_set.labels[i] << "." << endl;
        short result = predict(train_set, k, neighbors);
        if (+result == +test_set.labels[i])
        {
            success++;
        }
        else
        {
            errors++;
        }
        cout << "Success rate : " << (success * 100 / (success + errors)) << "%" << endl;
    }

    cout << success << " success and " << errors << " errors" << endl;
    cout << "Success rate : " << (success * 100 / (success + errors)) << "%" << endl;

    return 0;
}