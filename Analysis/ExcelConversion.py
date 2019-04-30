import sqlite3
import csv
import uuid
import sys
import os.path
import functools

class RaceConversion(object):
    """
    Handles the parsing of CSV copied/exported from Excel and the INSERT into a database
    CSV are assumed to be a single race spanning multiple precincts
    """
    def __init__(self, lines):
        self.lines = lines

        self.Race : str = None
        self.Precincts : List[str] = []
        self.Candidates : list[str] = []
        self.Results : List[tuple(str,str,int)] = [] # (candidate, precinct, total)

    def Parse1(self, precinct):
        # Format 1 *might* start with four lines that were the PDF headers. the fourth line contains the district
        # If the first cell is 'Summary Results Report', skip those three lines
        if self.lines[0][0] == 'Summary Results Report':
            precinct = self.lines[3][0]
            self.lines = self.lines[4:]
            
            # and it might contain statistics
            if self.lines[0][0] == 'STATISTICS':
                self.lines = self.lines[5:]

        self.Precincts.append(precinct)

        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Get the candidate names. These are on the same line as 'Registered Voters', etc.
        TotalVotesIdx = 1

        for row in self.lines[3:]:
            if row[0] in ('WRITE-IN', 'Total Votes Cast', 'Contest Totals', 'Summary Results Report'):
                break

            candidate, votes = row[:2]

            self.Candidates.append(candidate)
            
            try:
                total = int(votes)
            except:
                print("Non-integer vote count value ('%s') found for race '%s', precinct '%s', and candidate '%s'" % (votes, self.Race, precinct, candidate))
                continue

            self.Results.append((candidate, precinct, total))

        return precinct

    def Parse2(self):
        # fields in the CSV that are guaranteed and in the same place for every race.
        RegVoterIdx, TotalVotesIdx = 1, 2

        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is empty

        # Get the candidate names. These are on the same line as 'Registered Voters', etc.
        candidate_line = self.lines[2]
        candidates = [ (i, candidate_line[i]) for i in range(RaceConversion.TotalVotesIdx + 1, len(candidate_line)) if candidate_line[i].strip() != '']
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
                counts = [self.lines[p + 1][c], # normal
                    self.lines[p + 2][c], # absentee
                    self.lines[p + 3][c], # early voting
                    self.lines[p + 4][c]  # provisional
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

    def Parse3(self, precincts):
        self.Race = self.lines[0][0]
        
        self.Precincts = [name for (idx, name) in precincts]
        self.Candidates = self.lines[1]

        for p, precinct in precincts:
            candidate_line = self.lines[1]
            for c in range(len(candidate_line)):
                candidate = candidate_line[c]
                # precinct counts are on the third line
                total = self.lines[p][c]
                self.Results.append((candidate, precinct, int(total)))
        
    def Parse4(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is candidate names
        candidate_line = self.lines[1]
        candidates = [ (i, candidate_line[i]) for i in range(len(candidate_line)) if candidate_line[i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precinct names are listed in rows starting at index 4 and each take up five rows

        # Read from the fourth line to the last precinct; this format includes a Total, which is unnecessary
        precincts = [(i, self.lines[i][0]) for i in range(3, len(self.lines) - 1) if self.lines[i][0].strip() != '']
        self.Precincts = [name for (i,name) in precincts]


        for (p,precinct) in precincts:
            for (c, candidate) in candidates:
                # Total Votes is offset +3 from the candidate's name
                total = self.lines[p][c + 3]
                
                try:
                    total = int(total)
                except:
                    print("Non-integer vote count value ('%s') found for race '%s', precinct '%s', and candidate '%s'" % (total, self.Race, precinct, candidate))
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

    def Parse1(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """

        # Format 1 contains the precinct in column 0 as a precursor to all its data. Get the first one listed
        precinct = self.lines[3][0]

        # Format 1 starts with statistics on registered voter, ballots cast, voter turnout, etc.
        # The straight party stuff starts just before the first row that has 'VOTE 5' in the third column
        index = [i for i in range(len(self.lines)) if len(self.lines[i]) > 3 and self.lines[i][2] == 'VOTE %'][0]
        subset = self.lines[index:]

        # now remove all the counts and totals for straight party
        index = [i for i in range(len(subset)) if subset[i][0] == 'Contest Totals'][0]
        subset = subset[index + 1:]



        # Format 1 puts the string "STATISTICS" into the first column, unlike format 2 which is in the second
        index = [i for i in range(len(self.lines)) if self.lines[i][0] == 'STATISTICS'][0]
        subset = self.lines[index:]

        # find where each race ends
        contest_totals = [i for i in range(len(subset)) if subset[i][0] == 'Contest Totals']
        
        # the first race is straight party, so trim our data to everything past that
        subset = subset[contest_totals[0] + 1:]

        # shift the contest totals down since we've removed the straight party
        contest_totals = [None] + [i - contest_totals[0] for i in contest_totals[1:]]

        for i in range(len(contest_totals)-1):
            start, end = contest_totals[i], contest_totals[i + 1]
            race_subset = subset[start:end]

            race = RaceConversion(race_subset)
            precinct = race.Parse1(precinct)

            if race.Race == 'STRAIGHT PARTY':
                continue

            self.Races.append(race)

        race_count = len(self.Races)
        last_race = self.Races[race_count-1]

        # format 1 gives data first by precinct then by race, so we're left with a long list of 'Race' objects
        # that are for only a single precinct. combine these all so we have all precincts together
        race_names = list({r.Race for r in self.Races})
               
        consolidated :RaceConversion = []

        for race_name in race_names:
            races = [r for r in self.Races if r.Race == race_name]

            combined = RaceConversion([])
            combined.Race = race_name

            # simply combine all lists
            for r in races:
                combined.Candidates += r.Candidates
                combined.Precincts += r.Precincts
                combined.Results += r.Results

            # then get distinct values
            combined.Candidates = list(set(combined.Candidates))
            combined.Precincts = list(set(combined.Precincts))
            consolidated.append(combined)

        self.Races = consolidated



    def Parse2(self):
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
            race.Parse2()

            self.Races.append(race)

    def Parse3(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """

        # get the races, skipping the first two (statistics, straight party)
        race_line = self.lines[0]
        races = [(i, race_line[i]) for i in range(len(race_line)) if race_line[i].strip() != ''][2:]

        # dummy record
        races.append([None, None])

        precincts = [(i, self.lines[i][2]) for i in range(2, len(self.lines)) if self.lines[i][2] != 'COUNTY TOTALS']

        for i in range(len(races) - 1):
            r, race = races[i]
            r2, race2 = races[i + 1]

            # get the subset of text just for this race
            subset = [line[r:r2] for line in self.lines if line[2] != 'COUNTY TOTALS']

            r = RaceConversion(subset)
            r.Parse3(precincts)
            self.Races.append(r)

    def Parse4(self):
        """Format 4 is one race per sheet, so the ExcelConversion simply hands off parsing to Race
        """
        race = RaceConversion(self.lines)
        race.Parse4()
        self.Races = [race]

    def Dump(self):
        print("Races and candidates:")
        for race in self.Races:
            print(" - " + race.Race)
            for candidate in race.Candidates:
                print("    - " + candidate)
        print()

        r = self.Races[0]
        print("Precincts (from the first race, '%s'):" % r.Race)
        for precinct in r.Precincts:
            print("    " + precinct)
        print()

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
    #filename = '2018 General Election - Box Elder Precinct-Level Results.tsv' # format 2
    filename1 = '2018 General Election - Cache Precinct-Level Results - Format 1.csv'
    #filename2 = '2018 General Election - Beaver Precinct-Level Results - Format 2.csv'
    #filename3 = '2018 General Election - Grand Precinct-Level Results - Format 3.csv'
    #filename4 = '2018 General Election - Salt Lake Precinct-Level Results - Sheet 5 - Format 4.csv'

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        filename = sys.argv[1]

    converter = ExcelConversion(filename1)
    converter.Parse1()

    print("Parsing completed")
    converter.Dump()
