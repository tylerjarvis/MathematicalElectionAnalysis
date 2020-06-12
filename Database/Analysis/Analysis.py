import sqlite3

db = sqlite3.connect('../data entry/data.sqlite')
c = db.cursor()

c.execute("SELECT * FROM sqlite_master WHERE type = 'table'")
tables = c.fetchall()

c.execute("SELECT ElectionId FROM Election LIMIT 1")
electionId = c.fetchone()[0]

c.execute("""SELECT 
    Precinct.Name AS PrecinctName,
    Precinct.RegisteredVoterCount,
    Race.Description AS RaceDescription,
    SUM(Results.VotesReceived) AS TotalVotes
    FROM Precinct
    JOIN Results
        ON Precinct.PrecinctId = Results.PrecinctId
    JOIN Race
        ON Results.RaceId = Race.RaceId
    WHERE Race.ElectionId = ?
    GROUP BY Precinct.Name, Race.Description""", (electionId,))

races = c.fetchall()

print("Precinct | Registered Voters | Race | Votes In Race")
print("-" * 51)
for race in races:
    print(" | ".join(map(str, race)))

print(tables[0])