import sqlite3
import csv
import uuid
import sys, os.path


class RaceConversion(object):
    """
    Handles the parsing of CSV copied/exported from Excel and the INSERT into a database
    CSV are assumed to be a single race spanning multiple precincts
    """

    # fields in the CSV that are guaranteed and in the same place for every
    # race.
    RegVoterIdx, TotalVotesIdx = 1, 2

    def __init__(self, lines):
        self.lines = lines

        self.Race : str = None
        self.Precincts : List[str] = []
        self.Candidates : list[str] = []
        self.Results : List[tuple(str,str,int)] = []

    def Parse(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is empty

        # Get the candidate names. These are on the same line as 'Registered Voters', etc.
        candidates = [ (i, self.lines[2][i]) for i in range(RaceConversion.TotalVotesIdx + 1, len(self.lines[2])) if self.lines[2][i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precinct names are listed in rows starting at index 4 and each take up five rows

        # Get the index of the last row of all precincts. If the last row of our data is the last row of precincts,
        # then the empty string condition will fail, so check for that.
        precinct_end = [i for i in range(4, len(self.lines)) if self.lines[i][0].strip() in ('', 'Total')]
        precinct_end = precinct_end[0] if (len(precinct_end) > 0) else len(self.lines)
        precincts = [(i, self.lines[i][0]) for i in range(4, precinct_end, 5) if self.lines[i][0].strip() != '']
        self.Precincts = [name for (i,name) in precincts]

        for (p,precinct) in precincts:
            # The updated format (as of ~2019-04-13) lacks a 'Totals' column, so this has to be reconstructed from the
            # sum of Normal, Absentee, Early Voting, and Provisional, which are the four rows (in order) after the precinct name.
            # The 'Total Votes' column is +2 from the precinct name.
            for (c, candidate) in candidates:
                counts = [
                    self.lines[p+1][c], # normal
                    self.lines[p+2][c], # absentee
                    self.lines[p+3][c], # early voting
                    self.lines[p+4][c]  # provisional
                    ]

                # if thehre are no counts reported, skip this precinct+candidate
                if all([count == '' for count in counts]):
                    continue
                
                try:
                    total = sum(map(int, counts))
                except:
                    print("Non-integer vote count value found for race '%s', precinct '%s', and candidate '%s'" % (self.Race, precinct, candidate))
                    print("   Normal: %s" % counts[0])
                    print("   Absentee: %s" % counts[1])
                    print("   Early Voting: %s" % counts[2])
                    print("   Provisional: %s" % counts[3])
                    continue
                self.Results.append((candidate, precinct, total))
 
class ExcelConversion(object):
    def __init__(self, filename:str, delimiter:str=None):
        """
        @param: filename A CSV/TSV file to read in
        @param: delimiter Optional delimiter to use; if None, it is inferred from the extension
        """
        self.filename = filename
        if delimiter != None:
            self.delimiter = delimiter
        elif filename.endswith(".csv"):
            self.delimiter = ","
        elif filename.endswith(".tsv"):
            self.delimiter = "\t"
        else:
            raise "Delimiter unhandled"

        with open(self.filename, 'r') as fh:
            reader = csv.reader(fh, delimiter= self.delimiter)
            self.lines = [line for line in reader]

        self.Races : list[RaceConversion] = []

    def Parse(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """
        # Find the line with the race description
        # This 'cleaned' file puts the string "STATISTICS" into the second column, whereas a previous version put "TURN OUT" into the first
        # https://www.dropbox.com/ow/msft/edit/personal/Utah/UtahData/Election_Results/Precinct%20level/2018%20GE%20SOVC/Cleaned%20Data/Adrienne%20%26%20Annika%20conversions/2018%20General%20Election%20-%20Beaver%20Precinct-Level%20Results(1).xlsx?hpt_click_ts=1555170822997
        index = [i for i in range(len(self.lines)) if self.lines[i][1] == 'STATISTICS'][0]
        subset = self.lines[index:]

        # get the names and indexes of each race
        races = [(i, self.lines[index][i]) for i in range(len(self.lines[index])) if self.lines[index][i].strip() != '']

        # dummy record
        races.append((None, None))

        for i in range(len(races) - 1):
            (current_index, current_name) = races[i]
            (next_index, next_name) = races[i + 1]

            if current_name in ('STATISTICS', 'STRAIGHT PARTY'):
                continue

            # slice the list[][] and create a RaceConversion
            raceLines = [line[current_index:next_index] for line in subset]
            race = RaceConversion(raceLines)
            race.Parse()

            self.Races.append(race)

class ElectionDatabase(object):
    def __init__(self, filename):
        self.db = sqlite3.connect(filename)

    def GetCandidateId(self, name): return self.GetId('Candidate', 'Name', name)
    def GetDistrictId(self, desc): return self.GetId('District', 'Name', desc)
    def GetElectionId(self, desc): return self.GetId('Election', 'Description', desc)
    def GetPrecinctnId(self, name): return self.GetId('Precinct', 'Name', name)
    def GetRaceId(self, desc): return self.GetId('Race', 'Description', desc)

    def GetId(self, table, column, value):
        c = self.db.cursor()
        q = 'SELECT %sId FROM %s WHERE %s = ?' % (table, table, column)
        c.execute(q, (value,))
        result = c.fetchone()

        if type(result) == tuple:
            return result[0]
        
        # need to insert
        id = str(uuid.uuid4())
        q = 'INSERT INTO %s (%sId,%s) VALUES (?,?)' % (table, table, column)
        c.execute(q, (id, value))
        self.db.commit()
        return id

    def InsertResults(self, candidateId, precinctId, raceId, votes):
        c = self.db.cursor()
        # Check if we have results for this 3-tuple
        c.execute("SELECT * FROM Results WHERE CandidateId = ? AND PrecinctId = ? AND RaceId = ?", (candidateId, precinctId, raceId))
        result = c.fetchone()
        if type(result) == tuple: return

        q = 'INSERT INTO Results (CandidateId, PrecinctId, RaceId, VotesReceived) VALUES (?, ?, ?, ?)'
        c.execute(q, (candidateId, precinctId, raceId, votes))

    def CommitParsedResults(self, ec: ExcelConversion, election_name: str):
        electionId = self.GetElectionId(election_name)
        for rc in ec.Races:
            raceId = self.GetRaceId(rc.Race)
            for (candidate, precinct, votes) in rc.Results:
                # make sure candidate and precinct records exist
                candidateId = self.GetCandidateId(candidate)
                precinctId = self.GetPrecinctnId(precinct)

                self.InsertResults(candidateId, precinctId, raceId, votes)



if __name__ == '__main__':
    filename = '2018 General Election - Box Elder Precinct-Level Results.tsv'

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        filename = sys.argv[1]

    converter = ExcelConversion(filename)
    converter.Parse()

    print("Parsing completed")

    print("Races and candidates discovered:")
    for race in converter.Races:
        print(" - " + race.Race)
        for candidate in race.Candidates:
            print("    - " + candidate)
    print()

    r = converter.Races[0]
    print("Precincts discovered (from the first race, '%s'):" % r.Race)
    for precinct in r.Precincts:
        print("    " + precinct)
    print()

    #db = ElectionDatabase(r'C:\Users\adams\Source\Repos\T4DataEntry\T4DataEntry\bin\Debug\data.sqlite')
    #db.CommitParsedResults(converter, '2018 General Election')
    
