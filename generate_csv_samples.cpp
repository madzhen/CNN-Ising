

// Autocorrelation time in the 2-D Ising Model

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <deque> 
#include <string>        // to save values for autocorrelations
#include<sstream>
#include<iomanip>
#include <vector> 

using namespace std;
#define ISING_L 16
#define N_ALL L*L



std::random_device rd;
std::mt19937 generator(rd());
std::uniform_real_distribution<double> distribution(0.0, 1.0);



inline double std_rand()
{
	double rand_number = distribution(generator);
	return rand_number;
}

double J = +1;                  // ferromagnetic coupling
int Lx, Ly;                     // number of spins in x and y
int N;                          // number of spins
int **s;                        // the spins
double T;                       // temperature
double H = 0.1;                   // magnetic field
int flag_E = 0;                   // if flag is 1,stop autocorrelation accumulating.
int flag_m = 0;                   // if flag is 1,stop autocorrelation accumulating.
double w[17][3];                // Boltzmann factors
int hot_start = 0;               // choose the initial temperature




void computeBoltzmannFactors() {
	for (int i = -8; i <= 8; i += 4) {
		w[i + 8][0] = exp(-(i * J + 2 * H) / T);
		w[i + 8][2] = exp(-(i * J - 2 * H) / T);
	}
}

double eAv, mAv;                // accumulators to compute <e> and <m>
int nSave = 600;                 // values to save for autocorrelations
deque<double> eSave, mSave;      // saved energy and magnetization values
double *cee, *cmm;              // energy and magnetization correlation sums
int nCorr;                      // number of values accumulated in sums



void initializeCorrelations() {
	eAv = mAv = 0;
	eSave.clear();
	mSave.clear();

	if (cee != NULL) delete[] cee;
	if (cmm != NULL) delete[] cmm;
	cee = new double[nSave + 1];
	cmm = new double[nSave + 1];
	for (int i = 0; i <= nSave; i++)
		cee[i] = cmm[i] = 0;
	nCorr = 0;
}

int steps = 0;                  // steps so far


void initialize() {
	s = new int*[Lx];
	for (int i = 0; i < Lx; i++)
		s[i] = new int[Ly];


	if (hot_start)
	{
		for (int i = 0; i < Lx; i++)
			for (int j = 0; j < Ly; j++)
				s[i][j] = std_rand() < 0.5 ? +1 : -1;    // hot start

	}
	else
	{
		for (int i = 0; i < Lx; i++)
			for (int j = 0; j < Ly; j++)
				s[i][j] = -1; //COLD START
		for (int j = 0; j < Ly; j++)
			s[3][j] = 1;
	}

	computeBoltzmannFactors();
	steps = 0;
}

void initialize_for_Diff_T()
{
	if (hot_start)
	{
		for (int i = 0; i < Lx; i++)
			for (int j = 0; j < Ly; j++)
				s[i][j] = std_rand() < 0.5 ? +1 : -1;    // hot start

	}
	else
	{
		for (int i = 0; i < Lx; i++)
			for (int j = 0; j < Ly; j++)
				s[i][j] = -1; //COLD START
		for (int j = 0; j < Ly; j++)
			s[3][j] = 1;
	}
	computeBoltzmannFactors();
	steps = 0;
}

bool MetropolisStep() {

	// choose a random spin
	int i = int(Lx*std_rand());
	int j = int(Ly*std_rand());

	// find its neighbors using periodic boundary conditions
	int iPrev = i == 0 ? Lx - 1 : i - 1;
	int iNext = i == Lx - 1 ? 0 : i + 1;
	int jPrev = j == 0 ? Ly - 1 : j - 1;
	int jNext = j == Ly - 1 ? 0 : j + 1;

	// find sum of neighbors
	int sumNeighbors = s[iPrev][j] + s[iNext][j] + s[i][jPrev] + s[i][jNext];
	int delta_ss = 2 * s[i][j] * sumNeighbors;

	// ratio of Boltzmann factors
	double ratio = w[delta_ss + 8][1 + s[i][j]];
	if (std_rand() < ratio) {
		s[i][j] = -s[i][j];
		return true;
	}
	else return false;
}

double acceptanceRatio;

void oneMonteCarloStepPerSpin() {
	int accepts = 0;
	for (int i = 0; i < N; i++)
		if (MetropolisStep())
			++accepts;
	acceptanceRatio = accepts / double(N);
	++steps;
}

double magnetizationPerSpin() {
	int sSum = 0;
	for (int i = 0; i < Lx; i++)
		for (int j = 0; j < Ly; j++) {
			sSum += s[i][j];
		}
	return sSum / double(N);
}

double energyPerSpin() {
	int sSum = 0, ssSum = 0;
	for (int i = 0; i < Lx; i++)
		for (int j = 0; j < Ly; j++) {
			sSum += s[i][j];
			int iNext = i == Lx - 1 ? 0 : i + 1;
			int jNext = j == Ly - 1 ? 0 : j + 1;
			ssSum += s[i][j] * (s[iNext][j] + s[i][jNext]);
		}
	return -(J*ssSum + H*sSum) / N;
}

void accumulateCorrelations() {

	// calculate current energy and magnetization
	double e = energyPerSpin();
	double m = magnetizationPerSpin();


	// accumulate averages and correlation products
	if (eSave.size() == nSave) {   // if nSave values have been saved
		++nCorr;
		eAv += e;
		mAv += m;
		cee[0] += e * e;
		cmm[0] += m * m;
		deque<double>::const_iterator ie = eSave.begin(), im = mSave.begin();
		for (int i = 1; i <= nSave; i++) {
			cee[i] += *ie++ * e;
			cmm[i] += *im++ * m;
		}

		// discard the oldest values
		eSave.pop_back();
		mSave.pop_back();
	}

	// save the current values
	eSave.push_front(e);
	mSave.push_front(m);
}

double tau_e, tau_m;


void computeAutocorrelationTimes() {

	// energy correlation
	double av = eAv / nCorr;
	double c0 = cee[0] / nCorr - av * av;
	double auto_ke, auto_km;
	tau_e = 0;
	for (int i = 1; i <= nSave; i++)
	{
		auto_ke = abs((cee[i] / nCorr - av * av) / c0);
		if ((auto_ke > 0.015) && (flag_E == 0))
		{

			tau_e += auto_ke;
		}
		else
		{
			flag_E = 1;


		}

	}

	// magnetization correlation
	av = mAv / nCorr;
	c0 = cmm[0] / nCorr - av * av;
	tau_m = 0;
	for (int i = 1; i <= nSave; i++)
	{
		auto_km = abs((cmm[i] / nCorr - av * av) / c0);
		if ((auto_km > 0.015) && (flag_m == 0))
		{
	
			tau_m += auto_km;
		}
		else
		{
			flag_m = 1;
	

		}
	}
}



void Save_csv(deque<double>&auto_list, string _filename)
{
	int iterator_step = 0;
	deque<double>::iterator iterator_position;

	ofstream fout(_filename);

	if (!fout)
	{
		cout << "create file failed" << endl;

	}
	cout << "ok,pleas wait" << endl;
	for (iterator_position = auto_list.begin(); iterator_position != auto_list.end(); iterator_position++)
	{
		iterator_step++;

		fout << *iterator_position;
		if ((iterator_step) != auto_list.size())
		{
			fout << ",";
		}
	}
	fout.close();
}

string doubleConverToString(double d) {
	ostringstream os;
	if (os << setiosflags(ios::fixed) << setprecision(3) << d) return os.str();
	return "invalid conversion";
}

class Spin_conf
{
public:


	Spin_conf& operator=(int **spin_conf_s)
	{

		for (int i = 0; i < L; i++)
		{
			for (int j = 0; j < L; j++)
			{
				s[i*L + j] = spin_conf_s[i][j];
			}

		}
		return *this;
	}
	void output_s()
	{
		int sum = 0;
		for (int i = 0; i < N; i++)
		{
			sum += s[i];
		}
		cout << sum << endl;
	}

	int L = ISING_L;
	int N = 256;
	int s[256];

};



void Take_samples(vector<Spin_conf>&vec_McSamples, int N, string filename)
{
	vector<Spin_conf>::iterator iterator_position;

	ofstream fout(filename);

	if (!fout)
	{
		cout << "create file failed" << endl;
	}
	cout << "ok" << endl;
	for (iterator_position = vec_McSamples.begin(); iterator_position != vec_McSamples.end(); iterator_position++)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			fout << iterator_position->s[i];
			if (i != (N - 1))
			{
				fout << ",";
			}

		}
		fout << "\n";
	}
	fout.close();
}







int main(int argc, char *argv[]) {
	vector<Spin_conf> vec_McSamples;
	vector<Spin_conf>::iterator iterator_position;
	Spin_conf MC_stepN_conf;

	int hot_or_cold_start;
	cout << " If you want to choose hot start temperature,input 1 ,else 0 \n"
		<< " ---------------------------------------------------\n"
		<< " your input:   ";
	cin >> hot_or_cold_start;

	hot_start = hot_or_cold_start;

	cout << " Two-dimensional Ising Model - Autocorrelation times\n"
		<< " ---------------------------------------------------\n"
		<< " Enter number of spins L in each direction: ";
	cin >> Lx;
	Ly = Lx;
	N = Lx * Ly;
	double T1, T2;
	cout << " Enter starting temperature: ";
	cin >> T1;
	cout << " Enter ending temperature: ";
	cin >> T2;
	cout << " Enter number of temperature steps: ";
	int TSteps;
	cin >> TSteps;


	cout << " Enter number of Calculate_Auto_correlation steps: ";
	int Auto_Steps;
	cin >> Auto_Steps;


	cout << " Enter number of Sample steps: ";
	int Sample_steps;
	cin >> Sample_steps;

	initialize();
	ofstream file("auto_tau.csv");
	int thermSteps = int(0.2 * Auto_Steps);
	for (int i = 0; i <= TSteps; i++) {
		initialize_for_Diff_T();
		T = T1 + i * (T2 - T1) / double(TSteps);
		computeBoltzmannFactors();

		cout << "T= " << T << endl;
		for (int step = 0; step < thermSteps; step++)
		{
			oneMonteCarloStepPerSpin();

		}
		initializeCorrelations();
		for (int step = 0; step < Auto_Steps; step++) {
			oneMonteCarloStepPerSpin();
			accumulateCorrelations();
		}

		computeAutocorrelationTimes();

		cout << " T = " << T << "\ttau_e = " << tau_e
			<< "\ttau_m = " << tau_m << endl;
		file << T << ',' << tau_e << ',' << tau_m << '\n';
		cout << "now we will save the auto_corre" << endl;




		cout << "now we will start the samples for T= " << T<<' '<<endl;
		string Sample_filename = "Sample_T=" + doubleConverToString(T) + ".csv";


		int tau_m_stem = (int)(2*(tau_m));
		if (tau_m_stem==0)
		{
			tau_m_stem = 1;
		}

		for (int step=0;step<Sample_steps;step++)
		{

			while (tau_m_stem)
			{
				oneMonteCarloStepPerSpin();
				tau_m_stem--;
			}
			
			MC_stepN_conf = s;
			vec_McSamples.push_back(MC_stepN_conf);
		}


		Take_samples(vec_McSamples, N, Sample_filename);
		vec_McSamples.clear();



		flag_m = 0;
		flag_E = 0;


	}
	file.close();

}








