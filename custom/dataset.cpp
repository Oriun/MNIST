#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits.h>

using namespace std;

using pixel = unsigned char;
using matrix = array<pixel, 784>;
using image_set = vector<matrix>;
using label_set = vector<char>;

enum set
{
    train = false,
    test = true
};

struct dataset
{
    image_set imgs;
    label_set labels;
    dataset(image_set a, label_set b) : imgs(a), labels(b){};
};

string test_img_path = "./data/t10k-images-idx3-ubyte";
string test_label_path = "./data/t10k-labels-idx1-ubyte";
string train_img_path = "./data/train-images-idx3-ubyte";
string train_label_path = "./data/train-labels-idx1-ubyte";

void readImages(image_set &store, string path)
{
    ifstream file(path.c_str());
    if (file)
    {
        char a;
        unsigned int i(0);
        // Read header
        for (int j = 0; j <= 16; j++)
        {
            file.get(a);
        }
        // Read file
        do
        {
            int p = +a;
            if (p < 0)
            {
                p += 255;
            }

            int w = floor(i / 784);
            if (store.size() <= w)
            {
                store.push_back((matrix){});
            }
            store[w][i % 784] = static_cast<pixel>(a);

            i++;

        } while (file.get(a));
        cout << store.size() << endl;
    }
    else
    {
        cout << "Couldn't read file" << endl;
        throw runtime_error("Couldn't read file");
    }
}

void readLabels(label_set &store, string path)
{
    ifstream file(path.c_str());
    if (file)
    {
        char a;
        unsigned int i(0);
        // Read header
        for (int j = 0; j < 8; j++)
        {
            file.get(a);
        }
        // Read file
        while (file.get(a))
        {
            store.push_back(a);
        }
        cout << store.size() << endl;
    }
    else
    {
        cout << "Couldn't read file" << endl;
        throw runtime_error("Couldn't read file");
    }
}

dataset load(bool mode)
{
    image_set k;
    readImages(k, mode ? test_img_path : train_img_path);
    label_set l;
    readLabels(l, mode ? test_label_path : train_label_path);
    dataset data(k, l);

    return data;
}
