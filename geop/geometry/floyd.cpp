#include<iostream>
#include<vector>
#include<fstream>

using namespace std;

int main() {
  int N = 6890;
  float** D = new float*[N];
  for (int i = 0; i < N; i++) {
    D[i] = new float[N];
  }
  ifstream fin("adjacency");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      fin >> D[i][j];
    }
  }
  fin.close();
  for (int k = 0; k < N; k++) {
    cout << "k=" << k << endl;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if ((i == j) || (j == k) || (i == k)) {
          continue;
        }
        if (D[i][j] > D[i][k] + D[k][j]) {
          D[i][j] = D[i][k] + D[k][j];
        }
      }
    }
  }
  ofstream fout("floyd_results");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      fout << D[i][j] << " ";
    }
    fout << endl;
  }
  fout.close();
}
